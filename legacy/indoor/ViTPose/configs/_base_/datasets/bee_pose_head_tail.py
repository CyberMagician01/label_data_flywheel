dataset_info = dict(
    dataset_name='bee_pose_head_tail',
    paper_info=dict(
        author='Rodriguez et al.',
        title='Automated Video Monitoring of Unmarked and Marked Honey Bees',
        container='Frontiers in Computer Science',
        year='2021',
        homepage='https://github.com/piperod/beepose',
    ),
    keypoint_info={
        0: dict(name='head', id=0, color=[255, 80, 80], type='', swap=''),
        1: dict(name='abdomen_tip', id=1, color=[80, 180, 255], type='', swap=''),
    },
    skeleton_info={
        0: dict(link=('head', 'abdomen_tip'), id=0, color=[255, 255, 255])
    },
    joint_weights=[1.0, 1.0],
    sigmas=[0.05, 0.05],
)
