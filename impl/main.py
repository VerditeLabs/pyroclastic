import os

import download
import preprocess
import segment
import flatten
import detect
import postprocess

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

def main(create_zarr, writeable_zarr):
    """
    Pyroclasic is a "full scroll solver", i.e. it takes in a 3d voxel volume from
    a ct scan and downloads, preprocesses, segments, unrolls, ink detects, and
    postprocesses the volume.


 `   :return:`
    """
    zvol = download.ZVol(f"{ROOTDIR}", create_zarr, writeable_zarr)
    zvol.download( 'PHerc1667','20231107190228', 0, 1000)



if __name__ == '__main__':
    main(False, True)