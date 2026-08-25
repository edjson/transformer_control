function quanser_plant_physical(mode, host, port)
%QUANSER_PLANT_PHYSICAL  Hardware plant side. TCP client to cartpole_output.py.
%
%   quanser_plant_physical('dryrun')  real encoders read, motor NEVER energized
%   quanser_plant_physical('live')    motor energized
%
% Hardware only -- there is no 'sim' mode here on purpose, so there is no way
% to think you are on the rig when you are not. For simulation use
% quanser_plant.m.
%
% Start cartpole_output.py FIRST. The link is one-shot.
%
% Python sends force in Newtons and knows nothing about the plant.
% Volts, encoders and safety all live in this file.
%
% Run `clear functions` before every call. MATLAB caches this file and will
% not see edits made from the WSL side, and read_state holds persistent
% encoder offsets that must not carry over between runs.

if nargin < 1, mode = 'dryrun'; end
if nargin < 2, host = '127.0.0.1'; end
if nargin < 3, port = 5555; end

if strcmp(mode, 'sim')
    error('quanser_plant_physical:noSim', ...
          'No sim mode here. Use quanser_plant(''sim'').');
end
if ~ismember(mode, {'dryrun', 'live'})
    error('quanser_plant_physical:badMode', ...
          'mode must be ''dryrun'' or ''live'', got ''%s''.', mode);
end

%% ---- safety supervisor. Python cannot change any of this. ----
S.v_max    = 6.0;    % V   hard voltage ceiling
S.v_slew   = 40.0;   % V/s max rate of change of command
S.x_trip   = 0.35;   % m   |x| trip, inside the 0.407 rail half-length
S.xd_trip  = 1.2;    % m/s
S.thd_trip = 30.0;   % rad/s
S.wd       = 0.15;   % s   no valid command -> trip
S.tripped  = false;
S.reason   = '';

% Motor is dead in dryrun, so nothing unsafe can happen and hand-moving the
% cart trips x_trip immediately. Trips latch, which would kill the session.
if strcmp(mode, 'dryrun')
    S.x_trip = inf;  S.xd_trip = inf;  S.thd_trip = inf;  S.wd = inf;
end

%% ---- rig constants -- ALL UNVERIFIED. Replace after sysid_step. ----
%  When you measure Beq, update it in quanser_plant.m too or the two drift.
R.K_cart = 2.275e-5;        % m/count      UNVERIFIED
R.K_pole = 2*pi/4096;       % rad/count    UNVERIFIED
R.Rm = 2.6; R.Kt = 0.00767; R.Km = 0.00767;
R.Kg = 3.71; R.rmp = 6.35e-3; R.eta_g = 1.0; R.eta_m = 1.0;
R.Beq = 5.4;                % N.s/m        UNVERIFIED -- sysid_step replaces this
R.lp  = 0.85;               % derivative low-pass coefficient

%% ---- connect + handshake ----
t = tcpclient(host, port, 'Timeout', 5);
configureTerminator(t, "LF");
cfg = jsondecode(readline(t));
dt = cfg.dt;  N = cfg.n_steps;
fprintf('mode=%s  dt=%.4f  N=%d  (%.1f s)\n', mode, dt, N, N*dt);

%% ---- open board ----
% UNVERIFIED API names below. Board type, index and channel numbers depend on
% your QUARC version and hardware -- check the installed docs before running.
board = hil_open('q2_usb', '0');
cleanup = onCleanup(@() safe_shutdown(board)); %#ok<NASGU>

%% ---- zero the encoders ----
% Encoders read whatever count they held at power-up. Without this, x and th
% are offset by an arbitrary amount.
input('Centre the cart, let the pole hang still, then press Enter... ', 's');
st = read_state(board, R, dt, true);
fprintf('zeroed. x=%.4f m  th=%.1f deg (expect ~0 m, ~180 deg)\n', ...
        st(1), rad2deg(st(3)));

v_prev = 0;  t_last = tic;  t_cycle = tic;

for k = 0:N-1
    % ---- supervisor checks BEFORE anything is commanded ----
    [S, trip] = check_trips(S, st, toc(t_last));
    if trip
        writeline(t, jsonencode(struct('trip', true, 'reason', S.reason)));
        hil_write_analog(board, 0, 0.0);
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
    v_raw = v;
    v = max(-S.v_max, min(S.v_max, v));
    dv = S.v_slew * dt;
    v = max(v_prev - dv, min(v_prev + dv, v));
    v_prev = v;

    % ---- apply ----
    switch mode
        case 'dryrun'
            % motor stays dead. read only. v is computed for monitoring only.
            if mod(k, 4) == 0
                fprintf('x=%+.4f m  th=%+7.1f deg  xd=%+.3f  thd=%+.2f  v_would=%+.2f V\n', ...
                        st(1), rad2deg(wrapPi(st(3))), st(2), st(4), v_raw);
            end
        case 'live'
            hil_write_analog(board, 0, v);
    end

    % ---- pace the whole cycle, not just the write ----
    rem = dt - toc(t_cycle);
    if rem > 0
        pause(rem);
    elseif strcmp(mode, 'live')
        fprintf('period overrun %.1f ms\n', -rem*1e3);
    end
    dt_act = toc(t_cycle);  t_cycle = tic;
    st = read_state(board, R, dt_act, false);
end

hil_write_analog(board, 0, 0.0);
clear t
fprintf('plant done.\n');
end

% ============================ helpers ============================

function v = force_to_volts(F, xd, R)
A = R.eta_g*R.Kg*R.eta_m*R.Kt/(R.Rm*R.rmp);             % N per V
B = R.eta_g*R.Kg^2*R.eta_m*R.Kt*R.Km/(R.Rm*R.rmp^2);    % N per (m/s), back-EMF
v = (F + (B + R.Beq)*xd) / A;
end

function st = read_state(board, R, dt, first)
persistent x_prev th_prev xd_f thd_f c_cart0 c_pole0
c_cart = hil_read_encoder(board, 0);       % UNVERIFIED channel
c_pole = hil_read_encoder(board, 1);       % UNVERIFIED channel

if first || isempty(c_cart0)
    c_cart0 = c_cart;      % cart at centre when this runs
    c_pole0 = c_pole;      % pole hanging still when this runs
end

x     = (c_cart - c_cart0) * R.K_cart;
alpha = (c_pole - c_pole0) * R.K_pole;     % 0 = HANGING by construction, +CCW
th    = wrapPi(alpha + pi);                % model: 0 = UPRIGHT
% ^^ VERIFY IN DRYRUN. Push the pole toward +x and check th moves the way the
%    sim expects. If mirrored: th = wrapPi(pi - alpha) and flip K_cart's sign.

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
