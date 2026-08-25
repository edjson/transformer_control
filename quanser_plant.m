function quanser_plant(mode, host, port)
%QUANSER_PLANT  Plant side of the cartpole loop. TCP client to cartpole_output.py.
%
%   quanser_plant('sim')     RK4 model only, no hardware touched
%   quanser_plant('dryrun')  real encoders read, motor NEVER energized
%   quanser_plant('live')    motor energized
%
% Start cartpole_output.py FIRST. The link is one-shot.
%
% Python sends force in Newtons and knows nothing about the plant.
% Volts, encoders and safety all live in this file.

if nargin < 1, mode = 'sim'; end
if nargin < 2, host = '127.0.0.1'; end
if nargin < 3, port = 5555; end

%% ---- safety supervisor. Python cannot change any of this. ----
S.v_max    = 6.0;    % V   hard voltage ceiling
S.v_slew   = 40.0;   % V/s max rate of change of command
S.x_trip   = 0.35;   % m   |x| trip, inside the 0.407 rail half-length
S.xd_trip  = 1.2;    % m/s
S.thd_trip = 30.0;   % rad/s
S.wd       = 0.15;   % s   no valid command -> trip
S.tripped  = false;
S.reason   = '';

%% ---- rig constants -- ALL UNVERIFIED. Replace after sysid_step. ----
R.K_cart = 2.275e-5;        % m/count      UNVERIFIED
R.K_pole = 2*pi/4096;       % rad/count    UNVERIFIED
R.Rm = 2.6; R.Kt = 0.00767; R.Km = 0.00767;
R.Kg = 3.71; R.rmp = 6.35e-3; R.eta_g = 1.0; R.eta_m = 1.0;
R.Beq = 5.4;                % N.s/m        UNVERIFIED -- highest-value measurement
R.lp  = 0.85;               % derivative low-pass coefficient

%% ---- connect + handshake ----
t = tcpclient(host, port, 'Timeout', 5);
configureTerminator(t, "LF");
cfg = jsondecode(readline(t));
dt = cfg.dt;  N = cfg.n_steps;
mc = cfg.cart_mass;  mp = cfg.pole_mass;  L = cfg.pole_length;
fprintf('mode=%s  dt=%.4f  N=%d  mc=%.3f  mp=%.3f  L=%.4f\n', ...
        mode, dt, N, mc, mp, L);

board = [];
if ~strcmp(mode, 'sim')
    board = hil_open('q2_usb', '0');            % UNVERIFIED API name
    cleanup = onCleanup(@() safe_shutdown(board)); %#ok<NASGU>
end

% initial state: theta = 0 is UPRIGHT, pi is hanging. 0.3 = THETA0_OFFSET.
if strcmp(mode, 'sim')
    st = [0; 0; pi + 0.3; 0];
else
    st = read_state(board, R, dt, true);
end

v_prev = 0;  t_last = tic;

for k = 0:N-1
    % ---- supervisor checks BEFORE anything is commanded ----
    [S, trip] = check_trips(S, st, toc(t_last));
    if trip
        writeline(t, jsonencode(struct('trip', true, 'reason', S.reason)));
        if ~strcmp(mode, 'sim'), hil_write_analog(board, 0, 0.0); end
        fprintf('TRIP: %s\n', S.reason);
        break
    end

    % ---- send state, get force ----
    msg = struct('t', k*dt, 'x', st(1), 'xd', st(2), 'th', st(3), ...
                 'thd', st(4), 'status', mode, 'trip', false);
    writeline(t, jsonencode(msg));
    rep = jsondecode(readline(t));
    t_last = tic;
    if isfield(rep, 'stop') && rep.stop, break; end
    F = rep.u;

    % ---- force -> volts, clamp, slew limit ----
    v = force_to_volts(F, st(2), R);
    v = max(-S.v_max, min(S.v_max, v));
    dv = S.v_slew * dt;
    v = max(v_prev - dv, min(v_prev + dv, v));
    v_prev = v;

    % ---- apply ----
    switch mode
        case 'sim'
            F_eff = volts_to_force(v, st(2), R);
            st = rk4(st, F_eff, dt, mc, mp, L);
            st = track_limit(st, cfg.track_half);
        case 'dryrun'
            % motor stays dead. read only.
            st = read_state(board, R, dt, false);
        case 'live'
            hil_write_analog(board, 0, v);      % UNVERIFIED API name
            pause(dt);
            st = read_state(board, R, dt, false);
    end
end

if ~strcmp(mode, 'sim'), hil_write_analog(board, 0, 0.0); end
clear t
fprintf('plant done.\n');
end

% ============================ helpers ============================

function v = force_to_volts(F, xd, R)
A = R.eta_g*R.Kg*R.eta_m*R.Kt/(R.Rm*R.rmp);             % N per V
B = R.eta_g*R.Kg^2*R.eta_m*R.Kt*R.Km/(R.Rm*R.rmp^2);    % N per (m/s), back-EMF
v = (F + (B + R.Beq)*xd) / A;
end

function F = volts_to_force(v, xd, R)
A = R.eta_g*R.Kg*R.eta_m*R.Kt/(R.Rm*R.rmp);
B = R.eta_g*R.Kg^2*R.eta_m*R.Kt*R.Km/(R.Rm*R.rmp^2);
F = A*v - (B + R.Beq)*xd;
end

function d = dyn(st, F, mc, mp, L, g)
th = st(3);  thd = st(4);
c = cos(th);  s = sin(th);
temp  = (F + s*L*mp*thd^2) / (mc + mp);
thacc = (g*s - c*temp) / (L*(4/3 - mp*c^2/(mc + mp)));
xacc  = temp - (mp*L*thacc*c)/(mc + mp);
d = [st(2); xacc; thd; thacc];
end

function st = rk4(st, F, dt, mc, mp, L)
g = 9.81;
k1 = dyn(st,           F, mc, mp, L, g);
k2 = dyn(st+0.5*dt*k1, F, mc, mp, L, g);
k3 = dyn(st+0.5*dt*k2, F, mc, mp, L, g);
k4 = dyn(st+dt*k3,     F, mc, mp, L, g);
st = st + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function st = track_limit(st, half)
if st(1) >  half, st(1) =  half; st(2) = min(st(2), 0); end
if st(1) < -half, st(1) = -half; st(2) = max(st(2), 0); end
end

function st = read_state(board, R, dt, first)
persistent x_prev th_prev xd_f thd_f
c_cart = hil_read_encoder(board, 0);       % UNVERIFIED API name
c_pole = hil_read_encoder(board, 1);
x     = c_cart * R.K_cart;
alpha = c_pole * R.K_pole;                 % rig: 0 = HANGING, +CCW
th    = wrapPi(alpha + pi);                % model: 0 = UPRIGHT
% ^^ VERIFY THIS LINE IN DRYRUN BEFORE GOING LIVE. If mirrored, use
%    wrapPi(pi - alpha) and flip the sign on K_cart to match.
if first || isempty(x_prev)
    x_prev = x;  th_prev = th;  xd_f = 0;  thd_f = 0;
end
xd_raw  = (x - x_prev)/dt;
thd_raw = wrapPi(th - th_prev)/dt;
xd_f  = R.lp*xd_f  + (1 - R.lp)*xd_raw;
thd_f = R.lp*thd_f + (1 - R.lp)*thd_raw;
x_prev = x;  th_prev = th;
st = [x; xd_f; th; thd_f];
end

function [S, trip] = check_trips(S, st, since_cmd)
trip = S.tripped;
if trip, return; end
if abs(st(1)) > S.x_trip,   S.reason = 'x limit';       S.tripped = true; end
if abs(st(2)) > S.xd_trip,  S.reason = 'cart velocity'; S.tripped = true; end
if abs(st(4)) > S.thd_trip, S.reason = 'pole rate';     S.tripped = true; end
if since_cmd  > S.wd,       S.reason = 'watchdog';      S.tripped = true; end
trip = S.tripped;   % latching
end

function a = wrapPi(a)
a = atan2(sin(a), cos(a));
end

function safe_shutdown(board)
try, hil_write_analog(board, 0, 0.0); catch, end
try, hil_close(board); catch, end
end
