# Comparative Study of Traditional and Deep Learning Feature Detectors and Matchers for Land Vehicle Monocular Visual Odometry

This repository contains the code and data for our study, which compares traditional and deep learning-based feature detection and matching methods in the context of land vehicle monocular visual odometry (VO). The work evaluates performance in terms of computational efficiency, trajectory estimation accuracy, and robustness across various driving scenarios.

The code will be uploaded upon acceptance

## Abstract

Visual odometry (VO) enables autonomous navigation for land vehicles by estimating motion through visual data. This study compares classical detectors like SIFT and ORB with deep learning-based approaches such as SuperPoint and DISK. Similarly, traditional matchers like FLANN and SNN are evaluated alongside deep learning matchers such as ADALAM and LIGHTGLUE. The study emphasizes trade-offs between computational efficiency and accuracy under different motion dynamics and environmental conditions.

## Tested Data
### Scenario 1: Straight-Line Motion
- **Description**: Vehicle moves in a straight line at a consistent speed of ~30 km/h, covering a distance of 91 meters.
- **Visualization**:  
  ![Scenario 1](assets/gifs/sequence_1.gif)

### Scenario 2: Turning Motion
- **Description**: Vehicle slows down to ~10 km/h for a turn, then accelerates to ~15 km/h, covering a distance of 17.4 meters.
- **Visualization**:  
  ![Scenario 2](assets/gifs/sequence_2.gif)

### Scenario 3: Deceleration to Stop
- **Description**: Vehicle decelerates from ~32.5 km/h to 0, covering a distance of 37 meters.
- **Visualization**:  
  ![Scenario 3](assets/gifs/sequence_3.gif)

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/olaayman/MonocularVisualOdometry.git

