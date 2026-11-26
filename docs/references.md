# References

> **Note:** the course brief's own list of 6 references was not available while setting
> this up. The list below is a provisional set of the core works this pipeline builds on.
> Replace or extend it with the references from the AI512 project description.

1. **Deitke et al., "Objaverse: A Universe of Annotated 3D Objects", CVPR 2023.**
   Takeaway: a large open 3D asset collection with LVIS category labels and per-object
   licences. It is our source of bottle and can meshes; licences must be tracked (`LICENSES.csv`).

2. **Denninger et al., "BlenderProc2: A Procedural Pipeline for Photorealistic Rendering", JOSS 2023.**
   Takeaway: Blender-based procedural rendering with physics placement and built-in
   segmentation/COCO writers. It is the engine for our scenes and annotations.

3. **Proença & Simões, "TACO: Trash Annotations in Context for Litter Detection", arXiv:2003.06975, 2020.**
   Takeaway: a real litter dataset with segmentation masks of crushed/deformed waste "in the wild".
   It is our real-world test set for measuring the sim-to-real gap.

4. **Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017.**
   Takeaway: randomising textures, lighting and cameras makes the real world look like
   "just another variation". This motivates our random backgrounds, materials and lights.

5. **Tremblay et al., "Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization", CVPR Workshops 2018.**
   Takeaway: detectors trained on domain-randomised synthetic data with distractor objects
   can compete with real-data training. This motivates our distractor primitives and the A/B training design.

6. **Jocher et al., Ultralytics YOLO11, 2024 (software, https://github.com/ultralytics/ultralytics).**
   Takeaway: a small, fast single-stage detector (YOLO11n) with a standard training/evaluation
   CLI. It is our detector for the intact-vs-deformed training comparison.
