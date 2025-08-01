def __init__(self,
                xmem_ckpt,
                fbrs_ckpt,
                s2m_ckpt,
                **kwargs,
                ):
    super().__init__()

    self.save_hyperparameters()

    self.image_size = 512
    self.num_classes = 6
    self.nnunet_ckpt = nnunet_ckpt
    self.xmem_ckpt = xmem_ckpt
    self.fbrs_ckpt = fbrs_ckpt
    self.s2m_ckpt = s2m_ckpt

    self.idxs_in_memory = []

    config = VIDEO_INFERENCE_CONFIG.copy()
    config['num_objects'] =  self.num_classes
    config['size'] = self.image_size
    config['fbrs_ckpt'] = fbrs_ckpt
    config['s2m_ckpt'] = s2m_ckpt
    config['enable_long_term'] = True
    config['enable_long_term_count_usage'] = True

    self.xmem = XMem(config, xmem_ckpt, pretrained_key_encoder=False, pretrained_value_encoder=False).cuda().eval()

    if xmem_ckpt is not None:
        model_weights = torch.load(xmem_ckpt)
        self.xmem.load_weights(model_weights, init_as_zero_if_needed=True)

    self.processor = InferenceCore(self.xmem, config)
    self.processor.set_all_labels(list(range(1, num_classes + 1)))

def xmem_step(self, image, mask, idx):
    image = image.to(self.device)

    valid_labels = None

    with torch.cuda.amp.autocast(enabled=True):

        if mask is not None:
            mask = mask.to(self.device)
            valid_labels = range(1, self.num_classes + 1)
            self.processor.put_to_permanent_memory(image, mask)
            self.idxs_in_memory.append(idx)

        do_not_add_mask_to_memory = mask is not None
        prob = self.processor.step(image,
                                    mask,
                                    valid_labels=valid_labels,
                                    do_not_add_mask_to_memory=do_not_add_mask_to_memory,)

        out_mask = torch.argmax(prob, dim=0) - 1
        return out_mask