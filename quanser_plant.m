function quanser_plant(port, host)
%QUANSER_PLANT  Plant-side stand-in for a Quanser IP02-class linear cart.
%
%   quanser_plant()            connects to localhost:5555
%   quanser_plant(5555)
%   quanser_plant(5555, "172.20.1.5")
%
%   MATLAB half of the loop:
%
%       cartpole_output.py --{"cmd":"step","u":F_cmd}--> quanser_plant.m
%       cartpole_output.py <--{"x","xd","th","thd",...}-- quanser_plant.m
%
%   Owns everything on the HARDWARE side of the boundary:
%       * commanded force -> required motor voltage
%       * voltage saturation at the amplifier limit
%       * delivered force including back-EMF at the current cart speed
%       * equivalent viscous damping
%       * finite track: soft usable limit AND physical hard stop
%       * continuous-time integration between controller samples
%
%   TO GO TO REAL HARDWARE: replace the body of step_plant() with
%       hil_write_analog(card, ch, V);  s = hil_read_encoder(card, ...);
%   and delete cart_dyn/rk4. Nothing on the Python side changes.
%
%   Requires R2020b+ (tcpclient with configureTerminator).
%
%   Angle convention: theta = 0 is UPRIGHT, theta = pi is hanging down.
%   Positive x is right; positive force pushes the cart right.
%
%   NOTE: MATLAB will not notice edits to this file made from WSL. Run
%   "clear functions" before re-running after any change.

    if nargin < 1 || isempty(port), port = 5555; end
    if nargin < 2 || isempty(host), host = "localhost"; end

    P = plant_params();

    fprintf('[plant] connecting to %s:%d ...\n', host, port);
    c = tcpclient(host, port, "Timeout", 60, "ConnectTimeout", 30);
    configureTerminator(c, "LF");
    fprintf('[plant] connected.\n');

    dt       = 0.025;
    substeps = 10;
    s = [0; 0; pi; 0];      % [x; x_dot; theta; theta_dot]
    t = 0.0;

    while true
        try
            line = readline(c);
        catch ME
            fprintf('[plant] link closed (%s)\n', ME.message);
            break
        end
        if strlength(line) == 0
            continue
        end

        msg = jsondecode(line);
        cmd = string(msg.cmd);

        switch cmd
            case "reset"
                % Python owns the config -- take whatever it sends.
                if isfield(msg,'dt'),       dt        = msg.dt;       end
                if isfield(msg,'substeps'), substeps  = msg.substeps; end
                if isfield(msg,'x_limit'),  P.x_limit = msg.x_limit;  end
                if isfield(msg,'x_hard'),   P.x_hard  = msg.x_hard;   end
                if isfield(msg,'mc'),       P.mc      = msg.mc;       end
                if isfield(msg,'mp'),       P.mp      = msg.mp;       end
                if isfield(msg,'lp'),       P.lp      = msg.lp;       end
                if isfield(msg,'vmax'),     P.Vmax    = msg.vmax;     end
                if isfield(msg,'state0')
                    s = double(msg.state0(:));
                else
                    s = [0; 0; pi; 0];
                end
                t = 0.0;

                fprintf(['[plant] reset: dt=%.4f s, %d substeps, ' ...
                         'usable +/-%.3f m, hard +/-%.3f m, Vmax=%.1f V, ' ...
                         'mc=%.3f mp=%.3f lp=%.4f, th0=%.1f deg\n'], ...
                        dt, substeps, P.x_limit, P.x_hard, P.Vmax, ...
                        P.mc, P.mp, P.lp, rad2deg(s(3)));

                reply = struct('ok',1, 't',t, ...
                    'x',s(1), 'xd',s(2), 'th',s(3), 'thd',s(4), ...
                    'V',0.0, 'F',0.0, 'saturated',0, 'limit',0, ...
                    'Vmax',P.Vmax, ...
                    'Fmax_static', voltage_to_force(P.Vmax, 0, P), ...
                    'x_limit',P.x_limit, 'x_hard',P.x_hard);
                writeline(c, jsonencode(reply));

            case "step"
                F_cmd = double(msg.u);
                [s, t, V, F_act, sat, lim] = step_plant(s, t, F_cmd, dt, substeps, P);

                reply = struct('ok',1, 't',t, ...
                    'x',s(1), 'xd',s(2), 'th',s(3), 'thd',s(4), ...
                    'V',V, 'F',F_act, ...
                    'saturated',double(sat), 'limit',double(lim));
                writeline(c, jsonencode(reply));

            case "stop"
                writeline(c, jsonencode(struct('ok',1)));
                fprintf('[plant] stop received.\n');
                break

            otherwise
                writeline(c, jsonencode(struct('ok',0, ...
                    'err', sprintf('unknown cmd: %s', cmd))));
        end
    end

    clear c
    fprintf('[plant] shut down.\n');
end


% =========================================================================
%  ONE CONTROLLER PERIOD OF PLANT  --  replace with QUARC HIL calls
% =========================================================================
function [s, t, V, F_act, sat, lim] = step_plant(s, t, F_cmd, dt, substeps, P)

    % commanded force -> motor voltage, at the CURRENT cart speed
    V_req = force_to_voltage(F_cmd, s(2), P);

    % amplifier saturation: the constraint the policy never saw
    V   = max(min(V_req, P.Vmax), -P.Vmax);
    sat = abs(V_req) > P.Vmax + 1e-9;

    % integrate with VOLTAGE held (ZOH). Force varies inside the step as
    % back-EMF changes -- that is what actually happens on the rig.
    h = dt / substeps;
    for k = 1:substeps
        s = rk4(s, V, h, P);

        % physical hard stop: inelastic, cannot be passed
        if abs(s(1)) > P.x_hard
            s(1) = sign(s(1)) * P.x_hard;
            s(2) = 0;
        end
    end
    t = t + dt;

    F_act = voltage_to_force(V, s(2), P) - P.Beq * s(2);

    % soft limit: end of USABLE travel. Reported, not enforced.
    lim = abs(s(1)) >= P.x_limit - 1e-9;
end


% =========================================================================
%  motor / drivetrain model
% =========================================================================
function V = force_to_voltage(F, xdot, P)
    A = (P.eta_g * P.Kg * P.eta_m * P.Kt) / (P.Rm * P.rmp);
    B = (P.eta_g * P.Kg^2 * P.eta_m * P.Kt * P.Km) / (P.Rm * P.rmp^2);
    V = (F + B * xdot) / A;
end

function F = voltage_to_force(V, xdot, P)
    A = (P.eta_g * P.Kg * P.eta_m * P.Kt) / (P.Rm * P.rmp);
    B = (P.eta_g * P.Kg^2 * P.eta_m * P.Kt * P.Km) / (P.Rm * P.rmp^2);
    F = A * V - B * xdot;
end


% =========================================================================
%  dynamics -- numerically identical to the Python training model, so any
%  divergence you see is the HARDWARE effects, not a model change
% =========================================================================
function ds = cart_dyn(s, V, P)
    xd = s(2); th = s(3); thd = s(4);

    F = voltage_to_force(V, xd, P) - P.Beq * xd;

    ct  = cos(th);
    st  = sin(th);
    tot = P.mc + P.mp;

    temp  = (F + st * P.lp * P.mp * thd^2) / tot;
    thacc = (P.g * st - ct * temp) / (P.lp * (4/3 - P.mp * ct^2 / tot));
    xacc  = temp - (P.mp * P.lp * thacc * ct) / tot;

    ds = [xd; xacc; thd; thacc];
end

function s = rk4(s, V, h, P)
    k1 = cart_dyn(s,            V, P);
    k2 = cart_dyn(s + 0.5*h*k1, V, P);
    k3 = cart_dyn(s + 0.5*h*k2, V, P);
    k4 = cart_dyn(s + h*k3,     V, P);
    s  = s + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
end


% =========================================================================
%  DEFAULTS. Mechanical values are overwritten by the reset message --
%  Python owns those. The motor block below is NOT sent from Python and
%  is the one thing you must verify against your rig's manual.
% =========================================================================
function P = plant_params()
    P.g  = 9.81;

    P.mc = 0.57;        % cart mass, kg          (overwritten by reset)
    P.mp = 0.230;       % pole mass, kg          (overwritten by reset)
    P.lp = 0.3302;      % pivot-to-CoG, m        (overwritten by reset)

    P.x_limit = 0.814 / 2;   % usable travel     (overwritten by reset)
    P.x_hard  = 0.990 / 2;   % physical rail     (overwritten by reset)

    % --- motor / drivetrain, IP02-class nominal values -----------------
    % TODO: confirm against your unit. Wrong values here put the voltage
    % saturation in the wrong place, which is the whole point of this file.
    P.Rm    = 2.6;        % armature resistance, ohm
    P.Kt    = 0.00767;    % torque constant, N.m/A
    P.Km    = 0.00767;    % back-EMF constant, V.s/rad
    P.Kg    = 3.71;       % planetary gearbox ratio
    P.eta_g = 1.00;       % gearbox efficiency
    P.eta_m = 1.00;       % motor efficiency
    P.rmp   = 0.00635;    % motor pinion radius, m
    P.Beq   = 5.4;        % equivalent viscous damping at cart, N.s/m

    P.Vmax  = 10.0;       % amplifier limit, V   (overwritten by reset)
end
