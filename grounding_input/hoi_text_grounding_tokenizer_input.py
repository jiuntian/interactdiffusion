import torch as th


class HOIGroundingNetInput:
    def __init__(self):
        self.set = False

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the
        input only for the ground tokenizer.
        batch = {
            'subject_boxes': [[..],[..],..]],
            'object_boxes': [[..],[..],..]],
            'masks': ..,
            'subject_text_embeddings': [..],
            'object_text_embeddings': [..],
            'action_text_embeddings': [..]
        }
        """

        self.set = True

        subject_boxes = batch['subject_boxes']
        object_boxes = batch['object_boxes']
        masks = batch['masks']
        subject_positive_embeddings = batch["subject_text_embeddings"]
        object_positive_embeddings = batch["object_text_embeddings"]
        action_positive_embeddings = batch["action_text_embeddings"]
        # batch["image_embeddings"]

        self.batch, self.max_box, self.in_dim = subject_positive_embeddings.shape
        self.device = subject_positive_embeddings.device
        self.dtype = subject_positive_embeddings.dtype

        return {"subject_boxes": subject_boxes, "object_boxes": object_boxes,
                "masks": masks,
                "subject_positive_embeddings": subject_positive_embeddings,
                "object_positive_embeddings": object_positive_embeddings,
                "action_positive_embeddings": action_positive_embeddings}

    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference,
        please define the null input for the grounding tokenizer
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        subject_boxes = object_boxes = th.zeros(batch, self.max_box, 4, ).type(dtype).to(device)
        masks = th.zeros(batch, self.max_box).type(dtype).to(device)
        subject_positive_embeddings = object_positive_embeddings = action_positive_embeddings \
            = th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device)

        return {"subject_boxes": subject_boxes, "object_boxes": object_boxes,
                "masks": masks,
                "subject_positive_embeddings": subject_positive_embeddings,
                "object_positive_embeddings": object_positive_embeddings,
                "action_positive_embeddings": action_positive_embeddings}
