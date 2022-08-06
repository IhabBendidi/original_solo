from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "/projects/imagesets/imagenet/RawImages/val"
max_batch_size = 8


@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='cpu')

    return images, labels

pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=1)
pipe.build()
pipe_out = pipe.run()
images, labels = pipe_out
print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))

@pipeline_def
def shuffled_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir, random_shuffle=True, initial_fill=100)
    images = fn.decoders.image(jpegs, device='cpu')

    return images, labels
pipe = shuffled_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0, seed=1234)
pipe.build()

pipe_out = pipe.run()
images, labels = pipe_out
print(labels)
@pipeline_def
def rotated_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir, random_shuffle=True, initial_fill=21)
    images = fn.decoders.image(jpegs, device='cpu')
    rotated_images = fn.rotate(images, angle=10.0, fill_value=0)

    return rotated_images, labels

pipe = rotated_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0, seed=1234)
pipe.build()

pipe_out = pipe.run()
images, labels = pipe_out