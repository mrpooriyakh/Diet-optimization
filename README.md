# Dietary Optimization Project

## Overview
This project aims to optimize dietary patterns for the average Iranian individual using linear programming techniques. The focus is on achieving a balance between nutritional adequacy, affordability, cultural acceptance, and sustainability. This project integrates research-backed methodologies and mathematical modeling to address dietary and economic challenges.

## Features
- **Mathematical Modeling**: Utilizes linear programming to create optimal meal plans while adhering to daily nutritional requirements and budget constraints.
- **Customizable Parameters**: Adjustable dietary constraints and user preferences for flexible meal planning.
- **Data Analysis**: Analyzes nutritional values and cost efficiency to generate practical dietary recommendations.
- **Visualization**: Generates insightful plots for nutritional values and daily meal costs to support decision-making.

## Repository Structure
- `src/`: Contains the Python scripts, including the main optimization script (`Final_code.py`).
- `data/`: Holds datasets like nutritional information and cost data for meals.
- `docs/`: Includes detailed documentation and the final research paper (`Final-word.docx`).
- `outputs/`: Stores generated visual outputs such as `meals planned for each day.png` .
## How to Run
1. **Dependencies**: Install required Python libraries listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**: Place the dataset (e.g., `OR_project - Sheet1 (2).csv`) in the `data/` folder.

3. **Run the Code**:
   ```bash
   python src/Final_code.py
   ```

4. **View Outputs**:
   - Generated meal plans and summaries will be printed to the console.
   - Visualization plots (nutritional values and costs) will be saved in the `outputs/` directory.

## Goals
- To design an affordable, nutritionally adequate meal plan for the average Iranian individual.
- To ensure cultural relevance and sustainability in dietary recommendations.
- To provide a scalable solution for policy-makers and individuals in similar socio-economic contexts.

## Research Insights
This project is supported by an extensive review of dietary optimization techniques and their applications. The research paper (`Final-word.docx`) provides detailed insights into the methodology, data sources, and results achieved through this project.

### Key Findings
- **Caloric Efficiency**: Optimization reduced caloric intake to align with health recommendations while minimizing waste.
- **Protein Adequacy**: Improved protein intake compared to historical averages.
- **Cost Reduction**: Demonstrated significant cost savings while maintaining nutritional adequacy.

## Future Work
- Expanding the dataset to include broader dietary preferences and regional variations.
- Enhancing user interactivity for custom dietary constraints and goals.
- Incorporating environmental sustainability metrics into the optimization model.

## Acknowledgments
This project was developed by:
- **Pouriya Khodaparast**  
  [pooriyakh@aut.ac.ir](mailto:pooriyakh@aut.ac.ir)
- **Kimia Rasoulikeshvar**  
  [kimia.rasoulikeshvar@aut.ac.ir](mailto:kimia.rasoulikeshvar@aut.ac.ir)
- **Seyed Emad Hosseini**  
  [emad.hosseini@aut.ac.ir](mailto:emad.hosseini@aut.ac.ir)

For further details, refer to the research paper in the `docs/` directory.
