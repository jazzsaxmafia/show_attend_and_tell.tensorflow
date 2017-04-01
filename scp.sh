#################################################################
#   Copyright (C) 2017 Yancey Zhang. All rights reserved.
#														  
#	> File Name:        < scp.sh >
#	> Author:           < Yancey Zhang >		
#	> Mail:             < yanceyzhang2013@gmail.com >		
#	> Created Time:     < 2017/03/28 >
#	> Last Changed: 
#	> Description:
#################################################################

#!/bin/bash
# ssh断点续传
rsync -P --rsh=ssh feats.npy 104.199.232.77:/data/feats.npy

