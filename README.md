# nerfstudio-method-template
Template repository for creating and registering methods in Nerfstudio.

## File Structure
We recommend the following file structure:

```
├── my_method
│   ├── __init__.py
│   ├── my_config.py
│   ├── custom_pipeline.py [optional]
│   ├── custom_model.py [optional]
│   ├── custom_field.py [optional]
│   ├── custom_datamanger.py [optional]
│   ├── custom_dataparser.py [optional]
│   ├── ...
├── pyproject.toml
```

## Registering with Nerfstudio
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone or fork this repository and run the commands:

```
conda activate nerfstudio
cd nerfstudio-method-template/
pip install -e .
ns-install-cli
```

## Running The New Method
This repository creates a new Nerfstudio method named "method-template". Now, I renamed it to "deformable_gaussian" to fit the purpose. To train with it, run the command:
```
ns-train deformable_gaussian --data [PATH]
```

## Extract Image Metrics (PSNR, SSIM, LPIPS)
The following command creates the .json file containing the image metrics for the RtCW dataset. Find the path to the config.yml and run it.
```
ns-eval--load-config outputs/rtcw_sequence/deformable_gaussian/.../config.yml
```

## Monitoring
Monitoring the loss and convergence with respect to time.
```
tensorboard--logdir outputs/
```

## Rendering The Interpolated Video
On running this command, the renderer automatically selects a set of few random camera locations and creates a bezier joining these points and renders the video along the camera view path.
```
ns-render interpolate --load-config outputs/rtcw_sequence/deformable_gaussian/.../config.yml --output-path renders/dynamic_interpolation.mp4
'''
