MOD_ID = 'id'
MOD_RGB = 'rgb'
MOD_SS_DENSE = 'semseg_dense'
MOD_SS_CLICKS = 'semseg_clicks'
MOD_SS_SCRIBBLES = 'semseg_scribbles'
MOD_VALIDITY = 'validity_mask'

SPLIT_TRAIN = 'train'
SPLIT_VALID = 'val'

MODE_INTERP = {
    MOD_ID: None,
    MOD_RGB: 'bilinear',
    MOD_SS_DENSE: 'nearest',
    MOD_SS_CLICKS:  'sparse',
    MOD_SS_SCRIBBLES: 'sparse',
    MOD_VALIDITY: 'nearest',
}
