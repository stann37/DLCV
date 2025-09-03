from plyfile import PlyData

def count_gaussians_from_ply(ply_path):
    plydata = PlyData.read(ply_path)
    num_gaussians = plydata.elements[0].count
    print(f"Number of Gaussians: {num_gaussians}")
    return num_gaussians

# Usage 
ply_path = f"output/21/point_cloud/iteration_120000/point_cloud.ply" # 13414
num_gaussians = count_gaussians_from_ply(ply_path)