import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.widgets import Button, Slider
import os.path as osp
# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def test_plot_circle():
    # matplotlib.use('Agg')
    fig = plt.figure(figsize=(3, 3))
    plt.tight_layout()
    # ax = p3.Axes3D(fig)
    ax = fig.add_subplot(111, projection="3d")

    x_c = 1
    y_c = 0.1
    z_c = 1
    r = 2
    
    theta = np.linspace(0, 2 * np.pi, 300) # 300 points on the circle
    x = x_c + r * np.sin(theta)
    y = y_c + theta * 0.0
    z = z_c + r * np.cos(theta)
    import pdb; pdb.set_trace()
    ax.plot3D(x, y, z, color="red")
    plt.show()
    
    return

# Trajectory editing
def plot_3d_motion_static(save_path, kinematic_tree, joints, title, dataset, figsize=(5, 5), fps=120, radius=3,
                     vis_mode='default', gt_frames=[], traj_only=False, target_pose=None, kframes=[], obs_list=[]):
    
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(False)
    
    def init_2d():
        ax.set_xlim([-5, 5])
        ax.set_ylim([-3., 7])
        # print(title)
        fig.suptitle(title, fontsize=10)
        plt.gca().invert_yaxis()
        ax.grid(False)
    
    def update_value_sidebar(val):
        value = int(slider.val)
        # print(f'Selected Value: {value}')

        # Change color of points based on slider value
        # new_colors = np.random.rand(len(scatter_points), 3)  # Generate random colors
        # new_colors = np.random.rand(len(scatter_points.get_facecolors()), 3)  # Generate random colors
        # scatter_points.set_color(new_colors)

        # Change color of closest point to red
        colors_copy = colors.copy()
        colors_copy[value] = (1., 0, 0, 1.) #'red'
        sizes_copy = initial_sizes.copy()
        sizes_copy[value] = 20
        scatter_points.set_color(colors_copy)
        scatter_points.set_edgecolors(colors_copy)
        scatter_points.set_sizes(sizes_copy)
        plt.draw()

    def on_click_2d(event):

        if event.inaxes is ax:
            print(event.inaxes)
            print(ax)
            print(done_button.ax)
            print(event.inaxes is ax)
            print(event.inaxes is done_button.ax)
            
            # if len(selected_index) > 0:
            target_loc = np.array([event.xdata, event.ydata])
            target_locations.append(target_loc)
            cur_frame = int(slider.val)
            selected_index.append(cur_frame)
            # fig.canvas.mpl_disconnect(cid)
            print(f'From frame: {cur_frame}, x: {trajec[cur_frame, 0]}, y: {trajec[cur_frame, 1]}')
            print(f'Target at x: {event.xdata}, y: {event.ydata}')
            ax.scatter(target_loc[0], target_loc[1], color='red', s=3)
            ax.arrow(trajec[cur_frame, 0], trajec[cur_frame, 1], 
                     target_loc[0] - trajec[cur_frame, 0], 
                     target_loc[1] - trajec[cur_frame, 1],
                     overhang=0, head_width=0.1, head_length=0.2, color='green',
                     length_includes_head=True)
            plt.draw()
                
            # else:
            #     clicked_point = np.array([event.xdata, event.ydata])
            #     distances = np.linalg.norm(trajec - clicked_point, axis=1)
            #     closest_point_index = np.argmin(distances)
            #     closest_point = trajec[closest_point_index]
            #     selected_index.append(closest_point_index)
            #     print(f'Clicked at x: {event.xdata}, y: {event.ydata}')
            #     print(f'Closest data point idx: {closest_point_index}, x: {closest_point[0]}, y: {closest_point[1]}')

            #     # Change color of closest point to red
            #     colors_copy = colors.copy()
            #     colors_copy[closest_point_index] = (1., 0, 0, 1.) #'red'
            #     sizes_copy = initial_sizes.copy()
            #     sizes_copy[closest_point_index] = 10
            #     scatter_points.set_color(colors_copy)
            #     scatter_points.set_edgecolors(colors_copy)
            #     scatter_points.set_sizes(sizes_copy)
            #     plt.draw()
    
    def on_done_clicked(event):
        # fig.canvas.mpl_disconnect(cid)
        # Save figure
        # import pdb; pdb.set_trace()
        plt.savefig(osp.join(osp.dirname(save_path), "edit.png"), bbox_inches='tight')
        plt.close()  # Close the plot when button is clicked
    
    def on_add_clicked(event):
        # fig.canvas.mpl_disconnect(cid)
        print("Add clicked")

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    if target_pose is None:
        target_pose = np.zeros_like(data)

    # # preparation related to specific datasets
    # # For trajectory, we don't need to scale the data
    # if dataset == 'humanml':
    #     data *= 1.3  # scale for visualization
    #     target_pose *= 1.3
    #     obs_scale = [((loc[0] * 1.3, loc[1] * 1.3), rr * 1.3) for (loc, rr) in obs_list]
    # else:
    #     assert False, "Unknown dataset"

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_red = ["#ED5A37", "#E69E00", "#C75A39", "#FF7D00", "#EDB50E"]  # Generation color - keyframe
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    target_pose[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    ### Plot trajctory first
    # import pdb; pdb.set_trace()
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = fig.add_subplot(111)
    # Add space for sidebar
    plt.subplots_adjust(bottom=0.25)
    init_2d()
    selected_index = []
    target_locations = []
    # Add a slider
    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_ax, 'Frame', 0, 119, valinit=0, valfmt="%i")
    slider.on_changed(update_value_sidebar)
    ### Plot trajectory
    colors = ['black'] * len(trajec)
    initial_sizes = np.ones(len(trajec)) * 5
    scatter_points = ax.scatter(trajec[:, 0], trajec[:, 1], color=colors, s=initial_sizes)

    # Plot location to aviod (if any)
    for ((c_x, c_z), r) in obs_list:
        circ = Circle((c_x, c_z), r, facecolor='red', edgecolor='red', lw=1) # zorder=10
        ax.add_patch(circ)

    # Add a "done" button
    done_button = Button(plt.axes([0.7, 0.05, 0.1, 0.05]), 'Done')
    done_button.on_clicked(on_done_clicked)
    # Add a "add" button
    add_button = Button(plt.axes([0.81, 0.05, 0.1, 0.05]), 'Add')
    add_button.on_clicked(on_add_clicked)

    # Set up the event handler
    cid = fig.canvas.mpl_connect('button_press_event', on_click_2d)  # button_press_event
    plt.show()
    plt.close()

    ### Plot skeleton
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    # ax = p3.Axes3D(fig)
    ax = fig.add_subplot(111, projection="3d")
    init()


    # Data is root-centered in every frame
    # data_copy = data.copy()
    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    def plot_skel(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        # plot_obstacles(trajec[index])
        # plot_ground_target(trajec[index])

        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        # used_colors = colors_blue if index in gt_frames else colors
        # Now only use orange color. Blue color is used for ground truth condition
        # used_colors = colors_orange
        used_colors = colors_blue if index in gt_frames else colors # colors_red
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            # print("i = ", i, data[index, chain, 0], data[index, chain, 1], data[index, chain, 2])
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        if traj_only:
            ax.scatter(data[index, 0, 0], data[index, 0, 1], data[index, 0, 2], color=color)
        # Test plot trajectory
        # plot_trajectory(trajec[index])

        def plot_root_horizontal():
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 1]), trajec[:index, 1] - trajec[index, 1], linewidth=2.0,
                      color=used_colors[0])

        # plot_ref_axes(trajec[index])
        
        plot_root_horizontal()
        
        
        # plot_target_pose(target_pose, gt_frames, data_copy[index, 0, :], colors_blue, kinematic_tree)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    # import pdb; pdb.set_trace()

    # plot_skel(5)
    # Set up the event handler
    # fig.canvas.mpl_connect('button_press_event', on_click)

    # plt.show()
    plt.close()
    target_locations = target_locations[:len(selected_index)]
    # target_locations = [loc / 1.3 for loc in target_locations]
    return selected_index, target_locations


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[], traj_only=False, target_pose=None, kframes=[], obs_list=[],
                   clean_root=None):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
    
    def plot_trajectory(trajec_idx):
        ax.plot3D([0 - trajec_idx[0], 0 - trajec_idx[0]], [0.2, 0.2], [0 - trajec_idx[1], 1 - trajec_idx[1]], color="red") # (x,y,z)
    
    def plot_ref_axes(trajec_idx):
        '''
        trajec_idx contains (x,z) coordinate of the root of the current frame.
        Need to offset the reference axes because the plot is root-centered
        '''
        ax.plot3D([0 - trajec_idx[0], 0 - trajec_idx[0]], [0.2, 0.2], [0 - trajec_idx[1], 1 - trajec_idx[1]], color="red") # (x,y,z)
        ax.plot3D([0 - trajec_idx[0], 1 - trajec_idx[0]], [0.2, 0.2], [0 - trajec_idx[1], 0 - trajec_idx[1]], color="yellow") # (x,y,z)

    def plot_ground_target(trajec_idx):
        # kframes = [(30,  (0.0, 3.0)),
        #             (45,  (1.5, 3.0)),
        #             (60,  (3.0, 3.0)),
        #             (75,  (3.0, 1.5)),
        #             (90,  (3.0, 0.0)),
        #             (105, (1.5, 0.0)),
        #             (120, (0.0, 0.0))
        #             ]
        # import pdb; pdb.set_trace()
        pp = [(bb[0] * 1.3, bb[1] * 1.3) for (aa, bb) in kframes]
        for i in range(len(pp)):
            target = [pp[i][0] - trajec_idx[0], pp[i][0] - trajec_idx[0]], [0.0, 0.1], [pp[i][1] - trajec_idx[1], pp[i][1] - trajec_idx[1]]
            target = [ [round(a, 3), round(b, 3)] for (a, b) in target ]
            # import pdb; pdb.set_trace()
            target = np.array(target)
            # print(target)
            # print("--", [pp[i][0] - trajec_idx[0], pp[i][0] - trajec_idx[0]], [0.0, 0.1], [pp[i][1] - trajec_idx[1], pp[i][1] - trajec_idx[1]])
            
            ax.plot3D(*target, color="blue")

            # ax.plot3D([pp[i][0] - trajec_idx[0], pp[i][0] - trajec_idx[0]], [0.0, 0.1], [pp[i][1] - trajec_idx[1], pp[i][1] - trajec_idx[1]], color="blue") # (x,y,z)
        # print("---")

    def plot_obstacles(trajec_idx):
        for i in range(len(obs_scale)):
            x_c = obs_scale[i][0][0] - trajec_idx[0]
            y_c = 0.1
            z_c = obs_scale[i][0][1] - trajec_idx[1]
            r = obs_scale[i][1]
            # Draw circle
            theta = np.linspace(0, 2 * np.pi, 300) # 300 points on the circle
            x = x_c + r * np.sin(theta)
            y = y_c + theta * 0.0
            z = z_c + r * np.cos(theta)
            ax.plot3D(x, y, z, color="red") # linewidth=2.0

    def plot_target_pose(target_pose, frame_idx, cur_root_loc, used_colors, kinematic_tree):
        # The target pose is re-centered in every frame because the plot is root-centered
        # used_colors = colors_blue if index in gt_frames else colors
        for target_frame in frame_idx:
            for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                # print("i = ", i, data[index, chain, 0], data[index, chain, 1], data[index, chain, 2])
                ax.plot3D(target_pose[target_frame, chain, 0] - cur_root_loc[0],
                          target_pose[target_frame, chain, 1],
                          target_pose[target_frame, chain, 2] - cur_root_loc[2],
                          linewidth=linewidth, color=color)
    

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    if target_pose is None:
        target_pose = np.zeros_like(data)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
        target_pose *= 0.003
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
        target_pose *= 1.3
        obs_scale = [((loc[0] * 1.3, loc[1] * 1.3), rr * 1.3) for (loc, rr) in obs_list]
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization
        target_pose *= -1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    # ax = p3.Axes3D(fig)
    ax = fig.add_subplot(111, projection="3d")
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_red = ["#ED5A37", "#E69E00", "#C75A39", "#FF7D00", "#EDB50E"]  # Generation color - keyframe
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    target_pose[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    if clean_root is not None:
        xz_center = clean_root[:, [0, 2]]
    else:
        xz_center = trajec.copy()

    # Data is root-centered in every frame
    data_copy = data.copy()
    data[..., 0] -= xz_center[:, None, 0]
    data[..., 2] -= xz_center[:, None, 1]
    # data[..., 0] -= data[:, 0:1, 0]
    # data[..., 2] -= data[:, 0:1, 2]

    # Center first frame of target pose
    # target_pose[:, :, 0] -= data_copy[0:1, :, 0]
    # target_pose[:, :, 2] -= data_copy[0:1, :, 2]

    #     print(trajec.shape)

    def update(index):
        for line in ax.lines[:]:
            line.remove()
        # Clear existing collections
        for collection in ax.collections[:]:
            collection.remove()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - xz_center[index, 0], MAXS[0] - xz_center[index, 0], 
                     0, 
                     MINS[2] - xz_center[index, 1], MAXS[2] - xz_center[index, 1])

        # plot_obstacles(trajec[index])
        # plot_ground_target(trajec[index])
        plot_obstacles(xz_center[index])
        plot_ground_target(xz_center[index])

        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        # used_colors = colors_blue if index in gt_frames else colors
        # Now only use orange color. Blue color is used for ground truth condition
        # used_colors = colors_orange
        used_colors = colors_blue if index in gt_frames else colors # colors_red
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            # print("i = ", i, data[index, chain, 0], data[index, chain, 1], data[index, chain, 2])
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)
        if traj_only:
            ax.scatter(data[index, 0, 0], data[index, 0, 1], data[index, 0, 2], color=color)
        # Test plot trajectory
        # plot_trajectory(trajec[index])

        def plot_root_horizontal():
            ax.plot3D(trajec[:index, 0] - xz_center[index, 0], np.zeros_like(trajec[:index, 1]), 
                      trajec[:index, 1] - xz_center[index, 1], linewidth=2.0,
                      color=used_colors[0])

        # plot_ref_axes(trajec[index])
        
        # plot_root_horizontal()
        
        plot_target_pose(target_pose, gt_frames, data_copy[index, 0, :], colors_blue, kinematic_tree)

        plt.axis('off')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        ax.tick_params(axis='z', which='both', left=False, right=False, labelleft=False, labelright=False)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()
