% 2-D Kalman Filter model of object tracking
% William Turner

% motion params taken from: https://jov.arvojournals.org/article.aspx?articleid=2765454
object_speed_x = 3.35;
object_speed_y = -5.80;
pattern_speed_x = -5.196;
pattern_speed_y = -2.999;

% specify sampling rate and time window (s)
sampling_rate = 500; % samples per second
delta = 1/sampling_rate; % time step
time = 1.8; % time window of simulation
total_samples = time*sampling_rate;

% slow speed priors 
alpha = 0.9; % object motion
beta = 0.5; % pattern motion 

% specify state error (SDs) 
% used to make the process noise covariance matrix (Q)
% and the state covariance matrix (P)
position_p = 1e-17; % assume change in position is essentially perfectly described by speed x time. 
object_speed_p = 0.9; % assuming slight uncertainty in change in object speed (e.g., from bouncing etc.)
pattern_speed_p = 0.9; % assume same for pattern speed

% specify the measurement error (SDs)
% used to make the measurement covariance matrix (R)
% (note, for simplicities sake I have assumed equal measurement error. 
% however, the relative measurement errors can be adjusted without affecting the overall conclusion,
% so the specific values are somewhat arbitrary. What matters is that there is relatively 
% large uncertainty for the sensory measurements... which is reasonable for stimuli being viewed peripherally)
position_sigma = 4; % uncertainty of position signals
object_speed_sigma = 4; % uncertainty of object speed signals
pattern_speed_sigma = 4; % uncertainty of pattern speed signals

% system matrix
A = [1, delta, 0; 0, alpha, 0; 0, 0, beta];
A = blkdiag(A, A); % make system matrix 2-D (x AND y)

% observation matrix 
H = [1, 0, 0; 0, 1, 0; 0, 1, 1];
H = blkdiag(H, H); % make observation matrix 2-D

% Process noise covariance matrix
Q = diag([position_p, object_speed_p, pattern_speed_p]).^2;
Q = blkdiag(Q, Q); % make process noise covariance matrix 2-D

% Measurement noise covariance matrix
R = diag([position_sigma, object_speed_sigma, pattern_speed_sigma]).^2;

% Anisotropic motion noise
R_perpen = R;
R_perpen(3,3) = 0.125 * R(3,3); % this can only affect pattern motion
R_fixed_2D = blkdiag(R, R_perpen); % make measurement noise covariance matrix 2-D

% specify inputs
object_pos_x_series = ((1:total_samples) - 1) * (object_speed_x / sampling_rate);
object_pos_y_series = ((1:total_samples) - 1) * (object_speed_y / sampling_rate);
input_hat = [object_pos_x_series; repmat(object_speed_x, 1, total_samples); repmat(object_speed_x + pattern_speed_x, 1, total_samples);
             object_pos_y_series; repmat(object_speed_y, 1, total_samples); repmat(object_speed_y + pattern_speed_y, 1, total_samples)];

% Lognormal distribution
mu = -0.5;
sigma = 0.25;
steps = 200;
time_steps = linspace(0, 2.5, steps);
pdf = (1 ./ (time_steps .* sigma .* sqrt(2 * pi))) .* exp(-((log(time_steps) - mu).^2) ./ (2 * sigma.^2));
pdf(isnan(pdf)) = 0;

figure;

for flash = [1, 2]
    
    % set up number of loops (1 per gain modululation) and colors 
    if flash == 1
        loops = 0;         
    else
        loops = [0.00125, 0.005, 0.0125, 0.01875];
        colors = [0.99215686, 0.66459054, 0.39430988; ...
                 0.95893887, 0.4532872 , 0.12179931; ...
                 0.79607843, 0.26297578, 0.00607459; ...
                 0.52202999, 0.1621684 , 0.01507113];  % taken from python
    end
        
    for loop = loops
        
        if loop == 0
           color = [0.03137255, 0.18823529, 0.41960784]; % taken from python
        else
            color = colors(loop == loops, :);
        end
        
        gainMod = loop * pdf; % scale the pdf for to get the gain modulation function
        
        % Empty matrices for measurements...
        X_hat_minus = zeros(size(A, 1), total_samples);
        X_hat_plus = zeros(size(A, 1), total_samples);
        
        % ... and for covariances
        P_minus = zeros(size(A, 1), size(A, 1), total_samples);
        P_plus = zeros(size(A, 1), size(A, 1), total_samples);
        
        % initialise errors (exact values don't seem to matter too much)
        initial_P_plus = diag(diag(Q) ./ (1 - diag(A).^2));
        initial_P_plus(isinf(initial_P_plus)) = 1e-15;
        P_plus(:, :, 1) = initial_P_plus;
        
        Rotation_m = eye(6);

        % loop through timesteps
        for i = linspace(2, total_samples, total_samples - 1)
            
            Z = input_hat(:,i);   
  
            theta = cart2pol(Z(3), Z(6)); % get x and y speed for pattern (polar)
            theta_o = cart2pol(Z(2), Z(5)); % get x and y speed for object (polar)
            
            % Rotation  
            Rotation_m(3,3) = cos(theta);
            Rotation_m(3,6) = -sin(theta);
            Rotation_m(6,3) = sin(theta);
            Rotation_m(6,6) = cos(theta);

            Rotation_m(2,2) = cos(theta_o);
            Rotation_m(2,5) = -sin(theta_o);
            Rotation_m(5,2) = sin(theta_o);
            Rotation_m(5,5) = cos(theta_o);

            R = Rotation_m * R_fixed_2D * Rotation_m';
          
            %% PREDICT 
            % see eq. 7: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
            X_hat_minus(:,i) = A * X_hat_plus(:,i-1);
            P_minus(:,:,i) = (A * P_plus(:,:,i-1) * A') + Q;

            %% UPDATE %% 
            % see eqs. 18 and 19: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
            % K = P_minus(:, :, i) * H' * inv(H * P_minus(:, :, i) * H' + R);
            K = P_minus(:, :, i) * H' / (H * P_minus(:, :, i) * H' + R);

            % Additive gain modulation
            if i > 450 && i < (450 + length(time_steps)) 
                
                K = K + diag(repmat(gainMod(i-450), 1, 6));
                
            end
            
            % update means and covariances
            X_hat_plus(:,i) = X_hat_minus(:,i) + K * (Z - (H * X_hat_minus(:,i)));
            P_plus(:,:,i) = P_minus(:,:,i) - K * H * P_minus(:,:,i);
                    
        end
        
        % plot gain modulation function
        subplot(1,3,1)
        plot(gainMod, 'Color', color, 'LineWidth', 1.5)
        hold on
       
        % using approximate x and y offsets to position the simulated stimulus like Nakayama & Holcombe 
        % (unfortunately they don't give enough detail to perfectly replicate)
        xoffset = 5;
        yoffset = 10;
        
        subplot(1,3,flash+1)
        hold on 
        
        % plot one stimulus
        plot(xoffset + object_pos_x_series, yoffset + object_pos_y_series, 'LineWidth', 3, color = 'k')
        scatter(xoffset + squeeze(X_hat_plus(1,:)), yoffset + squeeze(X_hat_plus(4,:)), 0.5, color)    
        
        % plot the other stimulus (just mirror things)
        plot(-xoffset - object_pos_x_series, yoffset + object_pos_y_series, 'LineWidth', 3, color = 'k')
        scatter(-xoffset - squeeze(X_hat_plus(1,:)), yoffset + squeeze(X_hat_plus(4,:)), 0.5, color)
    
    end
    
    % specify plot aesthetics 
    if flash == 2
        title("Gain Modulation", fontsize = 15, fontweight = 'bold')  
        set(gca,'YColor','none')

    else
        title("No Modulation", fontsize = 15, fontweight = 'bold')
        ylabel('y position (dva)', fontsize = 15, fontweight = 'bold')
        yticks([-15, 0, 15])
    end
    
    xlim([-19, 19])
    xticks([-19,0,19])
    xlabel('x position (dva)', fontsize = 15, fontweight = 'bold')     
    ylim([-15, 15])
    ax = gca;
    ax.FontSize = 15; 
    box off
    set(gca,'color','none')

end

subplot(1,3,1)

% gain modulation plot aesthetics
yticks([0, 0.02, 0.04, 0.06])
title("Modulation Function", fontsize = 15, fontweight = 'bold')  % Add a title to the axes.
ylim([-0.0012, 0.065])
xlim([-10, 210])
xlabel('Time (ms)', fontsize = 15, fontweight = 'bold')  % Add an x-label to the axes.
ylabel('Δ Gain', fontsize = 15, fontweight = 'bold')  % Add a y-label to the axes.
ax = gca;
ax.FontSize = 15; 
box off
set(gca,'color','none')
