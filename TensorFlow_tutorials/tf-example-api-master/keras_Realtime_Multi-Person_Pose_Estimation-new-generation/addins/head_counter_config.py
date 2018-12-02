import sys
sys.path.append("..")

from config import Configs, GetConfig, COCOSourceConfig
import numpy as np

class HeadCounterConfig:

    def __init__(self):
        self.width = 368
        self.height = 368

        self.stride = 8

        self.parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank",
                      "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear", "HeadCenter"]
        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        self.parts += ["background"]
        self.num_parts_with_background = len(self.parts)

        leftParts, rightParts = self.ltr_parts(self.parts_dict)
        self.leftParts = leftParts
        self.rightParts = rightParts

        self.limb_from = \
            ['neck', 'Rhip', 'Rkne', 'neck', 'Lhip', 'Lkne', 'neck', 'Rsho', 'Relb', 'Rsho', 'neck', 'Lsho', 'Lelb', 'Lsho', 'neck', 'nose', 'nose', 'Reye', 'Leye']
        self.limb_to = \
            ['Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Rsho', 'Relb', 'Rwri', 'Rear', 'Lsho', 'Lelb', 'Lwri', 'Lear', 'nose', 'Reye', 'Leye', 'Rear', 'Lear']

        self.limb_from = [self.parts_dict[n] for n in self.limb_from]
        self.limb_to = [self.parts_dict[n] for n in self.limb_to]

        # this numbers probably copied from matlab they are 1.. based not 0.. based
        assert self.limb_from == [x - 1 for x in [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]]
        assert self.limb_to == [x - 1 for x in [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]]

        self.limbs_conn = list(zip(self.limb_from, self.limb_to))
        self.limbs_dict = dict(zip(self.limbs_conn, range(len(self.limbs_conn))))
        print(self.limbs_dict)

        self.paf_layers = 2 * len(self.limbs_conn)
        self.heat_layers = self.num_parts
        self.num_layers = self.paf_layers + self.heat_layers + 1

        self.paf_start = 0
        self.heat_start = self.paf_layers
        self.bkg_start = self.paf_layers + self.heat_layers

        # self.data_shape = (self.height, self.width, 3)     # 368, 368, 3
        self.mask_shape = (self.height // self.stride, self.width // self.stride)  # 46, 46
        self.parts_shape = (self.height // self.stride, self.width // self.stride, self.num_layers)  # 46, 46, 57

        class TransformationParams:

            def __init__(self):
                self.target_dist = 0.6;
                self.scale_prob = 1;  # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
                self.scale_min = 0.5;
                self.scale_max = 1.1;
                self.max_rotate_degree = 40.
                self.center_perterb_max = 40.
                self.flip_prob = 0.5
                self.sigma = 7.
                self.paf_thre = 8.  # it is original 1.0 * stride in this program

        self.transform_params = TransformationParams()


    def find_heat_layer(self, name):

        num = self.heat_start + self.parts_dict[name]
        return num

    def find_paf_layers(self, from_name, to_name):

        num_from = self.parts_dict[from_name]
        num_to = self.parts_dict[to_name]
        num = self.limbs_dict[ (num_from,num_to)]

        return (self.paf_start + 2*num, self.paf_start + 2*num + 1)


    @staticmethod
    def ltr_parts(parts_dict):
        # when we flip image left parts became right parts and vice versa. This is the list of parts to exchange each other.
        leftParts = [parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"]]
        rightParts = [parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"]]
        return leftParts, rightParts

class HeadCounterTrimmedConfig:

    def __init__(self):
        self.width = 368
        self.height = 368

        self.stride = 8

        self.parts = ["HeadCenter"]
        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        self.parts += ["background"]
        self.num_parts_with_background = len(self.parts)

        self.limbs_conn = []
        self.limbs_dict = {}

        self.leftParts = []
        self.rightParts = []

        self.paf_layers = 2 * len(self.limbs_conn)
        self.heat_layers = self.num_parts
        self.num_layers = self.paf_layers + self.heat_layers + 1

        self.paf_start = 0
        self.heat_start = self.paf_layers
        self.bkg_start = self.paf_layers + self.heat_layers

        # self.data_shape = (self.height, self.width, 3)     # 368, 368, 3
        self.mask_shape = (self.height // self.stride, self.width // self.stride)  # 46, 46
        self.parts_shape = (self.height // self.stride, self.width // self.stride, self.num_layers)  # 46, 46, 57

        class TransformationParams:

            def __init__(self):
                self.target_dist = 0.6;
                self.scale_prob = 1;  # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
                self.scale_min = 0.5;
                self.scale_max = 1.1;
                self.max_rotate_degree = 40.
                self.center_perterb_max = 40.
                self.flip_prob = 0.5
                self.sigma = 7.
                self.paf_thre = 8.  # it is original 1.0 * stride in this program

        self.transform_params = TransformationParams()


    def find_heat_layer(self, name):

        if name not in self.parts_dict: return None

        num = self.heat_start + self.parts_dict[name]
        return num

    def find_paf_layers(self, from_name, to_name):

        if from_name not in self.parts_dict: return (None,None)
        if to_name not in self.parts_dict: return (None,None)

        num_from = self.parts_dict[from_name]
        num_to = self.parts_dict[to_name]
        num = self.limbs_dict[ (num_from,num_to)]

        return (self.paf_start + 2*num, self.paf_start + 2*num + 1)



class COCOSourceHeadConfig(COCOSourceConfig):


    def __init__(self, hdf5_source):

        super().__init__(hdf5_source)

    def convert_mask(self, mask, global_config, joints = None):

        mask = super().convert_mask(mask, global_config, joints)

        HeadCenter = global_config.parts_dict['HeadCenter']

        HeadsCalculated = joints[:, HeadCenter, 2]

        if HeadsCalculated.shape[0]*2/3 > np.count_nonzero(HeadsCalculated < 2):

            # we have no idea there head is for more than 1/3 of pops. wipe whole layer.
            HeadCenterLayer = global_config.find_heat_layer('HeadCenter')
            mask[:,:, HeadCenterLayer] = 0.

            #print("nullified:", HeadsCalculated)
        else:
            #print("kept:", HeadsCalculated)
            pass




        return mask


    def convert(self, meta, global_config):

        old_joints = np.array(meta['joints'], dtype=np.float)
        retval = super().convert(meta, global_config)
        joints = retval['joints']

        assert old_joints is not joints

        HeadCenter = global_config.parts_dict['HeadCenter']
        Rear = self.parts_dict['Rear']
        Lear = self.parts_dict['Lear']
        Reye = self.parts_dict['Reye']
        Leye = self.parts_dict['Leye']

        both_ears_known = (old_joints[:, Rear, 2] < 2) & (old_joints[:, Lear, 2] < 2)
        Reye_Lear_known = (old_joints[:, Reye, 2] < 2) & (old_joints[:, Lear, 2] < 2)
        Leye_Rear_known = (old_joints[:, Leye, 2] < 2) & (old_joints[:, Rear, 2] < 2)

        joints[:, HeadCenter, 2] = 2. # otherwise they will be 3. aka 'never marked in this dataset'


        joints[Reye_Lear_known, HeadCenter, 0:2] = (          old_joints[Reye_Lear_known, Reye, 0:2] +
                                                              old_joints[Reye_Lear_known, Lear, 0:2]) / 2
        joints[Reye_Lear_known, HeadCenter, 2]   = np.minimum(old_joints[Reye_Lear_known, Reye, 2],
                                                              old_joints[Reye_Lear_known, Lear, 2])

        joints[Leye_Rear_known, HeadCenter, 0:2] = (          old_joints[Leye_Rear_known, Leye, 0:2] +
                                                              old_joints[Leye_Rear_known, Rear, 0:2]) / 2
        joints[Leye_Rear_known, HeadCenter, 2]   = np.minimum(old_joints[Leye_Rear_known, Leye, 2],
                                                              old_joints[Leye_Rear_known, Rear, 2])

        joints[both_ears_known, HeadCenter, 0:2] = (          old_joints[both_ears_known, Rear, 0:2] +
                                                              old_joints[both_ears_known, Lear, 0:2]) / 2
        joints[both_ears_known, HeadCenter, 2]   = np.minimum(old_joints[both_ears_known, Rear, 2],
                                                              old_joints[both_ears_known, Lear, 2])

        retval['joints'] = joints

        return retval





class MPIISourceHeadConfig:

    def __init__(self, hdf5_source):

        self.hdf5_source = hdf5_source
#        self.parts = ["HeadTop", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne",
#             "Rank", "Lhip", "Lkne", "Lank"]

        self.parts = [ "Rank", "Rkne", "Rhip", "Lhip", "Lkne", "Lank", "Pelvis", "Thorax", "neck", "HeadTop", "Rwri", "Relb", "Rsho", "Lsho", "Lelb", "Lwri"]

        # "(0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 10 - r wrist,
        # 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)"


        self.num_parts = len(self.parts)

        # for COCO neck is calculated like mean of 2 shoulders.
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))

        self.null_layers_cache = None

    def convert(self, meta, global_config):

        joints = np.array(meta['joints'])

        assert joints.shape[1] == len(self.parts)

        result = np.zeros((joints.shape[0], global_config.num_parts, 3), dtype=np.float)
        result[:,:,2]=3.  # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible

        for p in self.parts:

            if p in global_config.parts_dict:
                coco_id = self.parts_dict[p]
                global_id = global_config.parts_dict[p]
                result[:, global_id, :] = joints[:, coco_id, :]
            else:
                #assert p == "HeadTop" or p == "Pelvis" or p == "Thorax", p
                pass

        HeadCenterC = global_config.parts_dict['HeadCenter']
        neckC = self.parts_dict['neck']
        HeadTopC = self.parts_dict['HeadTop']

        # no head center, we calculate it as average of shoulders
        # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        both_parts_known = (joints[:, HeadTopC, 2]<2)  &  (joints[:, neckC, 2] < 2)

        result[~both_parts_known, HeadCenterC, 2] = 2. # otherwise they will be 3. aka 'never marked in this dataset'

        result[both_parts_known, HeadCenterC, 0:2] = (joints[both_parts_known, neckC, 0:2] +
                                                    joints[both_parts_known, HeadTopC, 0:2]) / 2
        result[both_parts_known, HeadCenterC, 2] = np.minimum(joints[both_parts_known, neckC, 2],
                                                                 joints[both_parts_known, HeadTopC, 2])

        meta['joints'] = result

        #for MPII scale=height/200, for COCO scale=height/368
        meta['scale_provided'] = [ x * 200/368 for x in meta['scale_provided'] ]

        return meta

    def convert_mask(self, mask, global_config, joints = None):

        mask = np.repeat(mask[:, :, np.newaxis], global_config.num_layers, axis=2)

        if self.null_layers_cache is None:

            Leye = global_config.find_heat_layer('Leye')
            Lear = global_config.find_heat_layer('Lear')
            Reye = global_config.find_heat_layer('Reye')
            Rear = global_config.find_heat_layer('Rear')
            nose = global_config.find_heat_layer('nose')

            neck_nose = global_config.find_paf_layers('neck','nose')
            nose_Reye = global_config.find_paf_layers('nose','Reye')
            nose_Leye = global_config.find_paf_layers('nose','Leye')
            Reye_Rear = global_config.find_paf_layers('Reye','Rear')
            Leye_Lear = global_config.find_paf_layers('Leye','Lear')

            self.null_layers_cache = (Leye,Lear,Reye,Rear,nose) + neck_nose + nose_Reye + nose_Leye + Reye_Rear + Leye_Lear
            self.null_layers_cache = [f for f in self.null_layers_cache if f is not None]
            print("Layers will be nullified: ", self.null_layers_cache)

        mask[:, :, self.null_layers_cache ] = 0.

        return mask

    def source(self):

        return self.hdf5_source


class PochtaSourceHeadConfig:

    def __init__(self, hdf5_source):

        self.hdf5_source = hdf5_source
        self.parts = [ "HeadCenter" ]
        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        self.mask_cache = None

    def convert(self, meta, global_config):

        joints = np.array(meta['joints'])

        assert joints.shape[1] == len(self.parts)

        result = np.zeros((joints.shape[0], global_config.num_parts, 3), dtype=np.float)
        result[:,:,2]=3.  # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible

        for p in self.parts:

            if p in global_config.parts_dict:
                pochta_id = self.parts_dict[p]
                assert pochta_id==0
                global_id = global_config.parts_dict[p]
                result[:, global_id, :] = joints[:, pochta_id, :]
            else:
                assert False

        meta['joints'] = result

        return meta

    def convert_mask(self, mask, global_config, joints = None):

        if self.mask_cache is not None:
            return self.mask_cache

        head_layer = global_config.find_heat_layer('HeadCenter')

        if global_config.num_parts==1:
            #background is ok
            print("Layers will be kept: ", head_layer, "background")
            self.mask_cache = np.ones(global_config.parts_shape, dtype=np.float)
        else:
            print("Layers will be kept: ", head_layer)
            self.mask_cache = np.zeros(global_config.parts_shape, dtype=np.float)
            self.mask_cache[:, :, head_layer ] = 1.

        return self.mask_cache


    def source(self):

        return self.hdf5_source


Configs["HeadCount"] = HeadCounterConfig
Configs["HeadTrim"] = HeadCounterTrimmedConfig



if __name__ == "__main__":

    # test it
    foo = GetConfig("HeadCount")


