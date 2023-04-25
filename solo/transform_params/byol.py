cifar_transform_dict = {
                0:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                1:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.05,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                2:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.0,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                3:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                4:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.0,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                5:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.4,
                                color_jitter_prob = 0.8,
                                min_scale = 0.4),
                6:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.2,
                                color_jitter_prob = 0.8,
                                min_scale = 0.4),
                7:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.05,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                8:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                color_jitter_prob = 0.8,
                                min_scale = 0.8),
                9:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.0,
                                color_jitter_prob = 0.8,
                                min_scale = 0.8),
                10:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.2,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                11:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.15,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                12:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.2,
                                color_jitter_prob = 0.8,
                                min_scale = 0.8),
                13:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.05,
                                color_jitter_prob = 0.8,
                                min_scale = 0.8),
                14:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.15,
                                color_jitter_prob = 0.8,
                                min_scale = 0.3),
                15:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                                color_jitter_prob = 0.8,
                                min_scale = 0.8),
                16:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.0,
                                color_jitter_prob = 0.8,
                                min_scale = 0.4),
                17:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.0,
                                color_jitter_prob = 0.8,
                                min_scale = 0.8),
                18:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.05,
                                color_jitter_prob = 0.8,
                                min_scale = 0.8),
                19:CifarTransform(cifar="cifar100",brightness=0.4,contrast=0.4,
                                saturation=0.2,
                                hue=0.4,
                                color_jitter_prob = 0.8,
                                min_scale = 0.7),
            }