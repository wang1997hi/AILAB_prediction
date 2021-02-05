import numpy as np

# 划线切割
# def reshape_patch(img_tensor, patch_size):
#     assert 5 == img_tensor.ndim
#     # b,l,c,h,w,
#     batch_size = np.shape(img_tensor)[0]
#     seq_length = np.shape(img_tensor)[1]
#     num_channels = np.shape(img_tensor)[2]
#     img_height = np.shape(img_tensor)[3]
#     img_width = np.shape(img_tensor)[4]
#     patch_tensor = img_tensor.reshape([batch_size, seq_length, num_channels,
#                                 patch_size, img_height//patch_size,
#                                 patch_size, img_width//patch_size])
#     patch_tensor = np.transpose(patch_tensor,(0,1,2,3,5,4,6))
#     patch_tensor = patch_tensor.reshape([batch_size, seq_length, num_channels*patch_size*patch_size,
#                                          img_height//patch_size,
#                                          img_width//patch_size])
#     return patch_tensor
#
# def reshape_patch_back(patch_tensor, patch_size):
#     assert 5 == patch_tensor.ndim
#     batch_size = np.shape(patch_tensor)[0]
#     seq_length = np.shape(patch_tensor)[1]
#     channels = np.shape(patch_tensor)[2]
#     patch_height = np.shape(patch_tensor)[3]
#     patch_width = np.shape(patch_tensor)[4]
#     img_channels = channels // (patch_size*patch_size)
#     img_tensor = patch_tensor.reshape([batch_size, seq_length, img_channels,
#                                        patch_size, patch_size,
#                                        patch_height, patch_width])
#     img_tensor = np.transpose(img_tensor, [0,1,2,3,5,4,6])
#     img_tensor = np.reshape(img_tensor, [batch_size, seq_length,img_channels,
#                                 patch_height * patch_size,
#                                 patch_width * patch_size])
#
#     return img_tensor



# 跳跃切割
def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    # b,l,c,h,w
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    num_channels = np.shape(img_tensor)[2]
    img_height = np.shape(img_tensor)[3]
    img_width = np.shape(img_tensor)[4]
    patch_tensor = img_tensor.reshape([batch_size, seq_length, num_channels,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size])
    patch_tensor = np.transpose(patch_tensor,(0,1,2,4,6,3,5))
    patch_tensor = np.reshape(patch_tensor, [batch_size, seq_length, patch_size*patch_size*num_channels,
                                  img_height//patch_size,img_width//patch_size])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    # b,l,c,h,w
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    channels = np.shape(patch_tensor)[2]
    patch_height = np.shape(patch_tensor)[3]
    patch_width = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size*patch_size)
    img_tensor = patch_tensor.reshape([batch_size, seq_length, img_channels,patch_size,patch_size,
                                       patch_height, patch_width])
    img_tensor = np.transpose(img_tensor, (0,1,2,5,3,6,4))
    img_tensor = np.reshape(img_tensor, [batch_size, seq_length,img_channels,
                                patch_height * patch_size,
                                patch_width * patch_size])
    return img_tensor
