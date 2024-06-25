from .img_label_dataset import ImageLabelDataset
from .source import DataSource


class ImagePairDataset(ImageLabelDataset):

    def load_label(self, data_id: str, data_source: DataSource, size):
        label = data_source.load_label(data_id)
        image = label['target_image']
        with self.random_context:
            data, crop_coord = self.bucket.crop_resize({"image": image}, size)
            image = data_source.process_label(data['image'])
        return {'label': {**label, 'target_image': image}}

    def __getitem__(self, index):
        (data_id, data_source), size = self.bucket[index]

        data = self.load_image(data_id, data_source, size)
        label = self.load_label(data_id, data_source, size)
        label = self.bucket.process_label(index, label)
        data.update(label)

        return data
