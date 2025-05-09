# 2-D Kalman Filter model of object tracking
# William Turner

# %% 

import numpy as np
import matplotlib.pyplot as plt
import warnings 

# suppress divide by zero warning which we can safely ignore. 
warnings.filterwarnings("ignore", category=RuntimeWarning)

def cart2pol(x, y):
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)

# motion params taken from: https://jov.arvojournals.org/article.aspx?articleid=2765454
object_speed_x = 3.35
object_speed_y = -5.80
pattern_speed_x = -5.196
pattern_speed_y = -2.999

# specify sampling rate and time window (s)
sampling_rate = 500; # samples per second
delta = 1/sampling_rate # time step
time = 1.8 # time window of simulation
total_samples = time*sampling_rate

# slow speed priors 
alpha = 0.9 # object motion
beta = 0.5 # pattern motion 

# specify state error (SDs) 
# used to make the process noise covariance matrix (Q)
# and the state covariance matrix (P)
position_p = 1e-17 # assume change in position is essentially perfectly described by speed x time. 
object_speed_p = 0.9 # assuming slight uncertainty in change in object speed (e.g., from bouncing etc.)
pattern_speed_p = 0.9 # assume same for pattern speed

# specify the measurement error (SDs)
# used to make the measurement covariance matrix (R)
# (Note, for simplicity I have assumed equal measurement error. 
# However, the relative measurement errors can be adjusted without affecting the overall conclusion,
# so the specific values are somewhat arbitrary. What matters is that there is relatively 
# large uncertainty for the sensory measurements... which is reasonable for stimuli being viewed peripherally)
position_sigma = 4 # uncertainty of position signals
object_speed_sigma = 4 # uncertainty of object speed signals
pattern_speed_sigma = 4 # uncertainty of pattern speed signals

# system matrix
A = np.array([[1, delta, 0], [0, alpha, 0], [0, 0, beta]])
A = np.block([[A, np.zeros_like(A)], [np.zeros_like(A), A]]) # make system matrix 2-D (x AND y)

# observation matrix 
H = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
H = np.block([[H, np.zeros_like(H)], [np.zeros_like(H), H]]) # make observation matrix 2-D

# process noise covariance matrix
Q = np.diag([position_p, object_speed_p, pattern_speed_p])**2
Q = np.block([[Q, np.zeros_like(Q)], [np.zeros_like(Q), Q]]) # make process noise covariance matrix 2-D

# measurement noise covariance matrix 
R = np.diag([position_sigma, object_speed_sigma, pattern_speed_sigma])**2

# implement anisotropic motion noise (see Kwon et al. supplement)
R_perpen = np.copy(R)
R_perpen[2,2] = 0.125 * R[2,2] # this can only effect pattern motion 
R_fixed_2D = np.block([[R, np.zeros_like(R)], [np.zeros_like(R), R_perpen]]) # make measurement noise covariance matrix 2-D

# specify inputs
object_pos_x_series = (np.arange(1, total_samples + 1) - 1) * (object_speed_x / sampling_rate)
object_pos_y_series = (np.arange(1, total_samples + 1) - 1) * (object_speed_y / sampling_rate)
input_hat = np.vstack((object_pos_x_series, np.repeat(object_speed_x, total_samples), np.repeat(object_speed_x + pattern_speed_x, total_samples),
                       object_pos_y_series, np.repeat(object_speed_y, total_samples), np.repeat(object_speed_y + pattern_speed_y, total_samples)))

# set up lognormal distribution (used as gain modulation function)
mu, sigma, steps = -0.5, 0.25, 200
time_steps = np.linspace(0.0001, 2.5, steps) # not endpoint of timesteps is 2.5 for scaling purposes but is otherwise meaningless... each timestep can be thought of as a ms  (+ added small constant to avoid divide by zero warning)
pdf = (1 / (time_steps * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(time_steps) - mu)**2 / (2 * sigma**2))
pdf[np.isnan(pdf)] = 0

fig, ax = plt.subplots(1,3,figsize=(15,4), dpi = 300)

for flash in [1,2]:
    
    # set up number of loops (1 per gain modululation) and colors 
    loops = [0] if flash == 1 else [0.00125, 0.005, 0.0125, 0.01875]
    
    colors = plt.cm.Blues([255]) if flash == 1 else plt.cm.Oranges([100,150,200,250])

    for x, loop in enumerate(loops):
        
        gainMod = loop * pdf # scale the pdf for to get the gain modulation function

        # set up empty matrices for measurements...
        X_hat_minus = np.array(np.zeros((np.shape(A)[0], int(total_samples))))
        X_hat_plus = np.array(np.zeros((np.shape(A)[0], int(total_samples))))
        
        # ... and for covariances
        P_minus = np.zeros((np.shape(A)[0], np.shape(A)[0], int(total_samples)))
        P_plus = np.zeros((np.shape(A)[0], np.shape(A)[0], int(total_samples)))

        # initialise errors (exact values don't seem to matter too much)
        initial_P_plus = np.array(np.diag(np.diag(Q)/(1-np.diag(A)**2)))
        initial_P_plus[initial_P_plus == float('+inf')] = 1e-15 
        P_plus[:, :, 0] = initial_P_plus
        
        Rotation_m = np.array(np.eye(6))

        # loop through timesteps
        for i in np.arange(1, int(total_samples)):
                        
            Z = input_hat[:,i]    
  
            theta = cart2pol(Z[2], Z[5]) # get x and y speed for pattern (polar)
            theta_o = cart2pol(Z[1], Z[4]) # get x and y speed for object (polar)

            Rotation_m[2,2] = np.cos(theta[1])
            Rotation_m[2,5] = -np.sin(theta[1])
            Rotation_m[5,2] = np.sin(theta[1])
            Rotation_m[5,5] = np.cos(theta[1])

            Rotation_m[1,1] = np.cos(theta_o[1])
            Rotation_m[1,4] = -np.sin(theta_o[1])
            Rotation_m[4,1] = np.sin(theta_o[1])
            Rotation_m[4,4] = np.cos(theta_o[1])

            R = np.dot(Rotation_m.dot(R_fixed_2D), Rotation_m.T)
                        
            ## PREDICT 
            # see eq. 7: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
            X_hat_minus[:,i] = A.dot(X_hat_plus[:,i-1])
            P_minus[:,:,i] = A.dot(P_plus[:,:,i-1]).dot(A.T) + Q

            ## UPDATE ## 
            # see eqs. 18 and 19: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
            K = np.dot(np.dot(P_minus[:,:,i], H.T), np.linalg.inv((np.dot(np.dot(H, P_minus[:,:,i]), H.T) + R)))

            ## Additive gain modulation
            if i >= 450 and i < 450 + len(time_steps):
                
                K += np.diag(np.repeat(gainMod[i-450], 6))                

            # Update means and covariances
            X_hat_plus[:,i] = X_hat_minus[:,i] + np.dot(K, (Z - np.dot(H, X_hat_minus[:,i])))
            P_plus[:,:,i] = P_minus[:,:,i] - np.dot(K, np.dot(H, P_minus[:,:,i]))   
            
        # plot gain modulation function
        ax[0].plot(gainMod, color = colors[x])
       
        # using approximate x and y offsets to position the simulated stimulus like Nakayama & Holcombe 
        # (unfortunately they don't give enough detail to perfectly replicate)
        xoffset = 5
        yoffset = 10
        
        # plot one stimulus
        ax[flash].plot(xoffset + object_pos_x_series, yoffset + object_pos_y_series, linewidth = 3, color = 'k', label="actual trajectory")
        ax[flash].scatter(xoffset + np.array(X_hat_plus[0,:]).flatten(), yoffset + np.array(X_hat_plus[3,:]).flatten(), s = 0.1, color = colors[x], label = "perceived trajectory")    
        
        # plot the other stimulus (just mirror things)
        ax[flash].plot(-xoffset - object_pos_x_series, yoffset + object_pos_y_series, linewidth = 3, color = 'k', label="actual trajectory")
        ax[flash].scatter(-xoffset - np.array(X_hat_plus[0,:]).flatten(), yoffset + np.array(X_hat_plus[3,:]).flatten(), s = 0.1, color = colors[x], label = "perceived trajectory")    
    
    # specify plot aesthetics 
    if flash == 2:
        ax[flash].set_title("Gain Modulation", fontsize = 15, fontweight = 'bold')  
        ax[flash].spines.left.set_visible(False)
        ax[flash].set_yticks([])

    else:
        ax[flash].set_title("No Modulation", fontsize = 15, fontweight = 'bold')
        ax[flash].set_ylabel('y position (dva)', fontsize = 15, fontweight = 'bold')
        ax[flash].set_yticks([-15, 0, 15])

    ax[flash].spines.bottom.set_bounds(-19, 19)
    ax[flash].set_xlim(-20, 19)
    ax[flash].set_xticks([-19,0,19])
    ax[flash].set_xlabel('x position (dva)', fontsize = 15, fontweight = 'bold')     

    ax[flash].spines.left.set_bounds(-15, 15)
    ax[flash].set_ylim(-16, 15)

    ax[flash].spines.right.set_visible(False)
    ax[flash].spines.top.set_visible(False)
    ax[flash].tick_params(axis='both', which='major', labelsize=15)

# gain modulation plot aesthetics 
ax[0].spines.left.set_bounds(0, 0.06)
ax[0].set_yticks([0, 0.02, 0.04, 0.06], ['0', '0.02', '0.04', '0.06'])
ax[0].set_xticks([0, 50, 100, 150, 200], ['0', '100', '200', '300', '400']) # remember sample rate is half
ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
ax[0].tick_params(axis='both', which='major', labelsize=15)

ax[0].set_title("Modulation Function", fontsize = 15, fontweight = 'bold')  # Add a title to the axes.
ax[0].set_ylim(-0.0012, 0.06)
ax[0].set_xlim(-10, 210)
ax[0].spines.bottom.set_bounds([0, 200])
ax[0].set_xlabel('Time (ms)', fontsize = 15, fontweight = 'bold')  # Add an x-label to the axes.
ax[0].set_ylabel(u'Δ Gain', fontsize = 15, fontweight = 'bold')  # Add a y-label to the axes.

# adjust spacing of subpanels 
box = ax[0].get_position()
box.x0, box.x1 = box.x0 - 0.02, box.x1 - 0.02
ax[0].set_position(box)

box = ax[1].get_position()
box.x0, box.x1 = box.x0 + 0.01, box.x1 + 0.01
ax[1].set_position(box)

# %% plot figure 2

# plot estimates and uncertainties over time
fig, ax = plt.subplots(2,3,figsize=(14,7), dpi = 300)
ax = ax.flatten()

# plot x and y position estimates + ground truth
ax[0].plot(np.arange(1, total_samples + 1), object_pos_x_series, color = 'blue', linestyle = '--', label = "true x position")
ax[0].plot(np.arange(1, total_samples + 1), object_pos_y_series, color = 'orange', linestyle = '--', label = "true y position")
ax[0].plot(np.arange(1, total_samples + 1), X_hat_plus[0,:], color = 'blue', label = "estimated x position")
ax[0].plot(np.arange(1, total_samples + 1), X_hat_plus[3,:], color = 'orange', label = "estimated y position")
ax[0].set_title("Position", fontsize = 15, fontweight = 'bold')
ax[0].set_ylabel('Position (dva)', fontsize = 15, fontweight = 'bold')
ax[0].set_ylim(-12 + (0.05 * -12), 12 + (0.05 * 12))
ax[0].spines.left.set_bounds(-12, 12)
ax[0].spines.bottom.set_bounds(0, 900)
ax[0].set_yticks([-12, -8, 0, 8, 12], ['', '-8', '0', '8', ''])
ax[0].set_xticks([0, 200, 400, 600, 800], ['0', '400', '800', '1200', '1600']) # remember sample rate is half, so adjust labels

# plot object velocity x and y estimates + ground truth
ax[1].plot(np.arange(1, total_samples + 1), np.repeat(object_speed_x, total_samples), color = 'blue', linestyle = '--', label = "true obj x speed")
ax[1].plot(np.arange(1, total_samples + 1), np.repeat(object_speed_y, total_samples), color = 'orange', linestyle = '--', label = "true obj y speed")
ax[1].plot(np.arange(1, total_samples + 1), X_hat_plus[1,:], color = 'blue', label = "estimated obj x speed")
ax[1].plot(np.arange(1, total_samples + 1), X_hat_plus[4,:], color = 'orange', label = "estimated object y speed")

# add vertical dashed lines at 450 and 650 (remember sample rate is half, so these are 900 and 1300 ms)
ax[1].axvline(x=450, color='black', linestyle='--', linewidth=0.5)
ax[1].axvline(x=650, color='black', linestyle='--', linewidth=0.5)

ax[1].set_title("Object Velocity", fontsize = 15, fontweight = 'bold')
ax[1].set_ylabel('Velocity (dva/s)', fontsize = 15, fontweight = 'bold')
ax[1].set_ylim(-9 + (0.05 * -9), 9 + (0.05 * 9))
ax[1].spines.left.set_bounds(-9, 9)
ax[1].spines.bottom.set_bounds(0, 900)
ax[1].set_yticks([-9, -6, 0, 6, 9], ['', '-6', '0', '6', ''])
ax[1].set_xticks([0, 200, 400, 600, 800], ['0', '400', '800', '1200', '1600']) 

# plot object pattern speed x and y estimates + ground truth
ax[2].plot(np.arange(1, total_samples + 1), np.repeat(pattern_speed_x, total_samples), color = 'blue', linestyle = '--', label = "true pattern x speed")
ax[2].plot(np.arange(1, total_samples + 1), np.repeat(pattern_speed_y, total_samples), color = 'orange', linestyle = '--', label = "true pattern y speed")
ax[2].plot(np.arange(1, total_samples + 1), X_hat_plus[2,:], color = 'blue', label = "estimated pattern x speed")
ax[2].plot(np.arange(1, total_samples + 1), X_hat_plus[5,:], color = 'orange', label = "estimated pattern y speed")
ax[2].axvline(x=450, color='black', linestyle='--', linewidth=0.5) # see comment above
ax[2].axvline(x=650, color='black', linestyle='--', linewidth=0.5)

ax[2].set_title("Pattern Velocity", fontsize = 15, fontweight = 'bold')
ax[2].set_ylim(-9 + (0.05 * -9), 9 + (0.05 * 9))
ax[2].spines.left.set_bounds(-9, 9)
ax[2].spines.bottom.set_bounds(0, 900)
ax[2].set_yticks([-9, -6, 0, 6, 9], ['', '-6', '0', '6', ''])
ax[2].set_xticks([0, 200, 400, 600, 800], ['0', '400', '800', '1200', '1600']) 

# plot positional uncertainties (convert variances to SDs by taking sqrt) over time
ax[3].plot(np.arange(1, total_samples + 1), np.sqrt(P_plus[0,0,:]), color = 'blue', label = "x position")
ax[3].plot(np.arange(1, total_samples + 1), np.sqrt(P_plus[3,3,:]), color = 'orange', label = "y position")
ax[3].axvline(x=450, color='black', linestyle='--', linewidth=0.5) # see comment above
ax[3].axvline(x=650, color='black', linestyle='--', linewidth=0.5)
ax[3].set_title("Position Uncertainty", fontsize = 15, fontweight = 'bold')
ax[3].set_ylabel('Uncertainty (dva)', fontsize = 15, fontweight = 'bold')
ax[3].set_ylim(-0.03 + (-0.1 * 0.03), 0.15 + (0.05 * 0.15))
ax[3].spines.left.set_bounds([-0.03, 0.15])
ax[3].set_yticks([-0.03, 0, 0.06, 0.12, 0.15], ['', '0', '0.06', '0.12', ''])
ax[3].spines.bottom.set_bounds(0, 900)
ax[3].set_xticks([0, 200, 400, 600, 800], ['0', '400', '800', '1200', '1600']) 

# plot object velocity uncertainties over time
ax[4].plot(np.arange(1, total_samples + 1), np.sqrt(P_plus[1,1,:]), color = 'blue', label = "x object speed")
ax[4].plot(np.arange(1, total_samples + 1), np.sqrt(P_plus[4,4,:]), color = 'orange', label = "y object speed")
ax[4].axvline(x=450, color='black', linestyle='--', linewidth=0.5) # see comment above
ax[4].axvline(x=650, color='black', linestyle='--', linewidth=0.5)
ax[4].set_title("Object Velocity Uncertainty", fontsize = 15, fontweight = 'bold')
ax[4].set_xlabel('\n Time (ms)', fontsize = 15, fontweight = 'bold')
ax[4].set_ylim(0.7 + (-0.05 * 0.7), 2.3 + (0.05 * 2.3))
ax[4].spines.left.set_bounds([0.7, 2.3])
ax[4].set_yticks([0.7, 1, 1.5, 2, 2.3], ['', '1', '1.5', '2', ''])
ax[4].spines.bottom.set_bounds(0, 900)
ax[4].set_xticks([0, 200, 400, 600, 800], ['0', '400', '800', '1200', '1600']) 

# plot some dummy data off screen for legend
l1, = ax[4].plot(0, -100, color = 'blue', linestyle = '--', label='true x')
l2, = ax[4].plot(0, -100, color = 'orange', linestyle = '--', label='true y')
l3, = ax[4].plot(0, -100, color = 'blue', label='estimated x')
l4, = ax[4].plot(0, -100, color = 'orange', label='estimated y')
ax[4].legend(handles=[l1, l2, l3, l4], labels=['true x', 'true y', 'estimated x', 'estimated y'],
             loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=False, shadow=False, ncol=4, 
             fontsize=15, frameon=False)

# plot pattern speed uncertainties over time
ax[5].plot(np.arange(1, total_samples + 1), np.sqrt(P_plus[2,2,:]), color = 'blue', label = "x pattern speed")
ax[5].plot(np.arange(1, total_samples + 1), np.sqrt(P_plus[5,5,:]), color = 'orange', label = "y pattern speed")
ax[5].axvline(x=450, color='black', linestyle='--', linewidth=0.5) # see comment above
ax[5].axvline(x=650, color='black', linestyle='--', linewidth=0.5)
ax[5].set_title("Pattern Velocity Uncertainty", fontsize = 15, fontweight = 'bold')
ax[5].spines.bottom.set_bounds(0, 900)
ax[5].set_ylim(0.85 + (-0.005 * 0.85), 1.15 + (0.025 * 1.15))
ax[5].spines.left.set_bounds(0.85, 1.15)
ax[5].set_yticks([0.85, 0.9, 1, 1.1, 1.15], ['', '0.9', '1', '1.1', ''])
ax[5].set_xticks([0, 200, 400, 600, 800], ['0', '400', '800', '1200', '1600']) 

# make some aethetic adjustments
# remove top and right spines
for a in ax: 
    a.spines.right.set_visible(False)
    a.spines.top.set_visible(False)

# shift axes 3-5 down 
for i in [3,4,5]:
    box = ax[i].get_position()
    box.y0, box.y1 = box.y0 - 0.1, box.y1 - 0.1
    ax[i].set_position(box)

# shift outer left and right axes (0, 2, 3, 5) 
for i in [0,3]:
    box = ax[i].get_position()
    box.x0, box.x1 = box.x0 - 0.02, box.x1 - 0.02
    
    ax[i].set_position(box)

for i in [2,5]:
    box = ax[i].get_position()
    box.x0, box.x1 = box.x0 + 0.02, box.x1 + 0.02

    ax[i].set_position(box)


# %%
