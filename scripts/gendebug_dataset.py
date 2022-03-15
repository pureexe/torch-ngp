import numpy as np
import os
import json
import argparse

parser = argparse.ArgumentParser(description='NeRF dataset only few nagle')
parser.add_argument('--dataset', type=str, default="data/nerf_synthetic/lego",  help='dataset location')
parser.add_argument('--source', type=str, default="test/20",  help='dataset location')
parser.add_argument('--angle', type=float, default=30.0,  help="angle to generate view")
args = parser.parse_args()

def cos_between_vectors(u,v):
    # @params u,v should be shape [n,3] where  n is number of angle
    # @return cos theta from a \cdot b = |a|||b| cos theta
    
    # make u into unit vector
    u_norm = np.linalg.norm(u,axis = 1).reshape((-1,1))
    u_norm = np.broadcast_to(u_norm, (u_norm.shape[0],3))
    u_unit = u / u_norm
    # make v into unit vector
    v_norm = np.linalg.norm(v,axis = 1).reshape((-1,1))
    v_norm = np.broadcast_to(v_norm, (v_norm.shape[0],3))
    v_unit = v / v_norm
    
    # dot product to get the cos(theta)
    # from a \cdot b = |a||b| cos theta
    u_dot_v = np.matmul(u_unit.reshape(-1,1,3), v_unit.reshape(-1,3,1))
    u_dot_v = np.squeeze(u_dot_v)
    return u_dot_v


def main():
    source_file, source_num = args.source.split('/')
    source_num = int(source_num)
    with open(os.path.join(args.dataset, "transforms_{}.json".format(source_file))) as f:
        data = json.load(f)
        loc = data["frames"][source_num]["transform_matrix"]
        loc = np.array(loc)[:3,3]
    
    for ftype in ["train","test","val"]:
        with open(os.path.join(args.dataset, "transforms_{}.json".format(ftype))) as fx:
            ori_data = json.load(fx)
            frames_loc = [ori_data["frames"][i]["transform_matrix"] for i in range(len(ori_data["frames"]))]
            positions = np.array(frames_loc)[:,:3,3]
            target_cast = np.broadcast_to(loc, (positions.shape[0],3))
            cos_angles = cos_between_vectors(positions, target_cast)
            keep_data = []  
            threshold_cos = np.cos(args.angle * np.pi / 180.0)
            for i in range(len(ori_data["frames"])):
                if  cos_angles[i] >= threshold_cos:
                    keep_data.append(ori_data["frames"][i])
            ori_data["frames"] = keep_data    
            print(len(ori_data["frames"]))
            
            with open(os.path.join(args.dataset, "transforms_{}_angles{}.json".format(ftype,args.angle)),'w') as fp:
                json.dump(ori_data, fp, indent=4)
        
        

if __name__ == "__main__":
    main()