default
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:18<00:00,  1.02s/it]
                   all        559        577      0.725      0.573      0.594      0.322
                 drone        122        138      0.667      0.435      0.467      0.186
                 plane        125        125      0.704      0.728       0.69      0.448
                  heli         63         63      0.895      0.905      0.942      0.603
                  bird        125        125      0.687      0.211      0.267     0.0969
                   uav        124        126      0.673      0.587      0.603      0.276

copypaste + small upds in config

                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:18<00:00,  1.00s/it]
                   all        559        577      0.733      0.461      0.609      0.333
                 drone        122        138      0.567       0.37      0.508      0.222
                 plane        125        125       0.57      0.584       0.57      0.357
                  heli         63         63      0.895       0.81      0.877      0.512
                  bird        125        125          1      0.072      0.536      0.294
                   uav        124        126      0.634      0.468      0.555      0.278
                   
copypaste + augs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:18<00:00,  1.00s/it]
                   all        559        577      0.732      0.571      0.678      0.349
                 drone        122        138      0.713      0.413      0.591      0.257
                 plane        125        125      0.662      0.704      0.697      0.391
                  heli         63         63      0.906      0.921      0.945      0.587
                  bird        125        125      0.676      0.184      0.447       0.18
                   uav        124        126      0.702      0.635       0.71       0.33
                   

copypaste + augs + wloss bbox bigger coef                  
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:17<00:00,  1.01it/s]
                   all        559        577      0.739      0.262      0.496      0.247
                 drone        122        138      0.778      0.101      0.439       0.21
                 plane        125        125      0.667       0.16      0.401      0.214
                  heli         63         63      0.724      0.667      0.708      0.323
                  bird        125        125          1      0.008      0.504      0.302
                   uav        124        126      0.528      0.373      0.427      0.184
                   
copypaste + augs + wwloss focal loss bigger coef
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 18/18 [00:18<00:00,  1.00s/it]
                   all        559        577      0.754      0.514      0.636      0.349
                 drone        122        138      0.797      0.341      0.553      0.278
                 plane        125        125      0.551      0.608      0.575      0.348
                  heli         63         63      0.891      0.905      0.932      0.575
                  bird        125        125       0.85      0.136      0.499      0.244
                   uav        124        126      0.682      0.579       0.62      0.301

copypaste + augs + multiscale (+ batch 8)
worse

yolo small version + copypaste + augs (+ batch 8)
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 35/35 [00:47<00:00,  1.35s/it]
                   all        559        577      0.771      0.383      0.568       0.27
                 drone        122        138      0.742      0.355      0.572      0.249
                 plane        125        125      0.587      0.512      0.482      0.202
                  heli         63         63      0.927      0.603      0.782      0.409
                  bird        125        125          1      0.056      0.528      0.285
                   uav        124        126      0.598      0.389      0.478      0.206