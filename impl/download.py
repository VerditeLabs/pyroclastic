import zarr
import requests
import os
import io
import tifffile
import numpy as np
from PIL import Image
from numcodecs import Blosc
import shutil


the_index = {
    'PHerc0332': {
        '20231027191953': {'depth': 22941, 'height': 9414, 'width': 9414, 'ext':'jpg', 'url': 'https://dl.ash2txt.org/community-uploads/james-darby/PHerc0332/volumes_masked/20231027191953_jpg/'},
        '20231117143551': {'depth': 9778,  'height': 3550, 'width': 3400},
        '20231201141544': {'depth': 22932, 'height': 9414, 'width': 9414},
    },
    'PHerc1667': {
        '20231107190228': {'depth': 26391, 'height': 7960, 'width': 8120, 'ext': 'jpg', 'url': 'https://dl.ash2txt.org/community-uploads/james-darby/PHerc1667/volumes_masked/20231107190228_jpg/'},
        '20231117161658': {'depth': 11174, 'height': 3340, 'width': 3440},
    },
    'Scroll1': {
        '20230205180739': {'depth': 14376, 'height': 7888, 'width': 8096, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/volumes_masked/20230205180739/'},
    },
    'Scroll2': {
        '20230210143520': {'depth': 14428, 'height': 10112, 'width': 11984, 'ext': 'tif', 'url': 'https://dl.ash2txt.org/full-scrolls/Scroll2.volpkg/volumes_masked/20230210143520/'},
        '20230212125146': {'depth': 1610, 'height': 8480, 'width': 11136},
    },
}

USER = os.environ.get('SCROLLPRIZE_USER')
PASS = os.environ.get('SCROLLPRIZE_PASS')

def _download(url):
    response = requests.get(url, auth=(USER, PASS))
    if response.status_code == 200:
        filedata = io.BytesIO(response.content)
        if url.endswith('.tif'):
            with tifffile.TiffFile(filedata) as tif:
                data = tif.asarray()
                if data.dtype == np.uint16:
                    return ((data >> 8) & 0xf0).astype(np.uint8)
                else:
                    raise
        elif url.endswith('.jpg'):
            data = np.array(Image.open(filedata))
            return data & 0xf0
        elif url.endswith('.png'):
            data = np.array(Image.open(filedata))
            return data
    else:
        raise Exception(f'Cannot download {url}')


class ZVol:
    def __init__(self, path, create=False, write=True):
        synchronizer = zarr.ProcessSynchronizer(f'{path}/vesuvius.sync')
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.BITSHUFFLE)
        exists = os.path.exists(f'{path}/vesuvius.zarr')

        if exists and not create:
            mode = 'r+' if write else 'r'
            root = zarr.open(f'{path}/vesuvius.zarr', mode=mode)
            self.root = root
            return

        if create and exists:
            shutil.rmtree(f'{path}/vesuvius.zarr')
            shutil.rmtree(f'{path}/vesuvius.sync')

        if not create:
            raise ValueError(f'{path} does not exist, pass create=True to create a zarr')
        root = zarr.group(store=f'{path}/vesuvius.zarr', synchronizer=synchronizer)

        root.create_group('PHerc0332')
        root.create_group('PHerc1667')
        root.create_group('Scroll1')
        root.create_group('Scroll2')

        for scroll, id in [['PHerc0332', '20231027191953'],
                           ['PHerc0332', '20231117143551'],
                           ['PHerc0332', '20231201141544'],
                           ['PHerc1667', '20231107190228'],
                           ['PHerc1667', '20231117161658'],
                           ['Scroll1', '20230205180739'],
                           ['Scroll2', '20230210143520'],
                           ['Scroll2', '20230212125146'], ]:
            _ = the_index[scroll][id]
            n, h, w = _['depth'], _['height'], _['width']
            root[scroll].zeros(id,
                               shape=(n, h, w),
                               chunks=(256, 256, 256),
                               dtype='u1',
                               compressor=compressor,
                               synchronizer=synchronizer, order='C')
            root[scroll].zeros(id + '_downloaded', shape=n, dtype='u1', synchronizer=synchronizer)
        self.root = root

    def download(self, scroll, id, start, end):
        """downloads 2d tiff slices from the vesuvius challenge and converts them into a
        zarr array"""

        depth = the_index[scroll][id]['depth']
        url = the_index[scroll][id]['url']
        ext = the_index[scroll][id]['ext']

        for x in range(start, end):
            filename = f"{x:0{len(str(depth))}d}.{ext}"
            if self.root[scroll][id+'_downloaded'][x] == 1:
                print(f"skipped {url}{filename}")
                continue
            print(f"Downloading {url}{filename}")
            data = _download(url + filename)
            print(f"Downloaded {url}{filename}")
            self.root[scroll][id][x,:,:] = data
            self.root[scroll][id+'_downloaded'][x] = 1
            print(f"wrote {url}{filename}")

    def chunk(self, scroll, id, start, size):
        self.download(scroll,id,start[0],start[0]+size[0])
        return self.root[scroll][id][start[0]: start[0] + size[0], start[1]: start[1] + size[1], start[2]: start[2] + size[2]]