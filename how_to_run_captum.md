## Instructions on how to get visualizations for the captum outputs

```bash
python captum_visualization.py \
    --temperature 8 \
    --sq_attr_path <the_path_to_your_sq_attr.npy> \
    --output_name <a_name_for_the_output_file>
```

For `--temperature`, the lower it is, the visualization is more "concentrated" to certain words. 

For example: you can run the following

```bash
python captum_visualization.py \
    --temperature 8 \
    --sq_attr_path sq_attr_raw.npy \
    --output_name test_name
```

This will generates an html visualization like this:

<p align="center">
  <img src='./static/captum.png' style="width: 100%; height: auto;"/>
</p>