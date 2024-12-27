import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class NutritionalConstraints:
    """Defines daily nutritional requirements and meal-specific ratios"""
    DAILY_REQUIREMENTS = {
        'calories_min': 1800,
        'calories_max': 2500,
        'protein_min': 45,
        'fat_max': 85,
        'carbs_max': 300
    }
    
    MEAL_RATIOS = {
        'breakfast': 0.25,
        'lunch': 0.35,
        'dinner': 0.25,
        'snack': 0.15
    }

class MealPlanner:
    def __init__(self):
        self.solver = SolverFactory('glpk', executable='C:\\glpk\\w64\\glpsol.exe')
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.meal_types = ['breakfast', 'lunch', 'dinner', 'snack']
        self.constraints = NutritionalConstraints()

    def preprocess_data(self, df):
        """Clean and prepare the dataset"""
        processed_df = df.copy()
        
        # Clean cost column
        processed_df['cost (rial)'] = processed_df['cost (rial)'].apply(
            lambda x: float(str(x).replace('$', '').replace(',', '')) 
            if pd.notnull(x) and str(x).strip() != '' else 0
        )
        
        # Clean nutritional columns
        nutritional_cols = ['Calories', 'Protein (g)', 'Fat (g)', 'Carbs (g)']
        for col in nutritional_cols:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
        
        return processed_df

    def create_meal_model(self, df, meal_type, used_meals):
        """Create optimization model for a single meal"""
        # Filter available meals
        available_df = df[
            (df['Category'] == meal_type) & 
            (~df.index.isin(used_meals))
        ].copy()
        
        if len(available_df) == 0:
            print(f"No available meals for {meal_type}")
            return None, None
            
        # Create model
        model = ConcreteModel()
        
        # Sets
        model.FOODS = Set(initialize=available_df.index)
        model.PORTIONS = Set(initialize=range(1, 6))  # 1-5 portions (100g-500g)
        
        # Variables
        model.select = Var(model.FOODS, model.PORTIONS, domain=Binary)
        
        # Objective - Minimize cost
        model.objective = Objective(
            expr=sum(available_df.loc[f, 'cost (rial)'] * p * model.select[f,p]
                    for f in model.FOODS for p in model.PORTIONS),
            sense=minimize
        )
        
        # Constraints
        ratio = self.constraints.MEAL_RATIOS[meal_type]
        
        # Select exactly one food-portion combination
        model.one_meal = Constraint(
            expr=sum(model.select[f,p] for f in model.FOODS for p in model.PORTIONS) == 1
        )
        
        # Nutritional constraints
        model.calories_min = Constraint(
            expr=sum(available_df.loc[f, 'Calories'] * p * model.select[f,p]
                    for f in model.FOODS for p in model.PORTIONS) 
            >= self.constraints.DAILY_REQUIREMENTS['calories_min'] * ratio
        )
        
        model.calories_max = Constraint(
            expr=sum(available_df.loc[f, 'Calories'] * p * model.select[f,p]
                    for f in model.FOODS for p in model.PORTIONS) 
            <= self.constraints.DAILY_REQUIREMENTS['calories_max'] * ratio
        )
        
        model.protein = Constraint(
            expr=sum(available_df.loc[f, 'Protein (g)'] * p * model.select[f,p]
                    for f in model.FOODS for p in model.PORTIONS) 
            >= self.constraints.DAILY_REQUIREMENTS['protein_min'] * ratio
        )
        
        return model, available_df

    def solve_daily_meals(self, df, day, used_meals):
        """Plan all meals for a single day"""
        daily_meals = []
        daily_cost = 0
        daily_nutrition = {'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0}
        
        for meal_type in self.meal_types:
            print(f"Planning {meal_type} for {day}...")
            
            model, available_df = self.create_meal_model(df, meal_type, used_meals)
            if model is None:
                continue
                
            results = self.solver.solve(model)
            
            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):
                
                # Extract selected meal
                for f in model.FOODS:
                    for p in model.PORTIONS:
                        if value(model.select[f,p]) > 0.5:  # Binary variable threshold
                            meal_data = {
                                'meal_type': meal_type,
                                'food_id': f,
                                'dish_name': available_df.loc[f, 'Dish Name'],
                                'portion': p * 100,
                                'calories': available_df.loc[f, 'Calories'] * p,
                                'protein': available_df.loc[f, 'Protein (g)'] * p,
                                'fat': available_df.loc[f, 'Fat (g)'] * p,
                                'carbs': available_df.loc[f, 'Carbs (g)'] * p,
                                'cost': available_df.loc[f, 'cost (rial)'] * p
                            }
                            
                            daily_meals.append(meal_data)
                            used_meals.add(f)
                            daily_cost += meal_data['cost']
                            
                            # Update daily nutrition
                            for nutrient in ['calories', 'protein', 'fat', 'carbs']:
                                daily_nutrition[nutrient] += meal_data[nutrient]
            else:
                print(f"Could not find optimal solution for {meal_type}")
        
        return {
            'day': day,
            'meals': daily_meals,
            'total_cost': daily_cost,
            'nutrition': daily_nutrition
        }

    def plan_week(self, df):
        """Generate complete weekly meal plan"""
        df = self.preprocess_data(df)
        weekly_plan = []
        used_meals = set()
        
        for day in self.days:
            daily_plan = self.solve_daily_meals(df, day, used_meals)
            weekly_plan.append(daily_plan)
            print(f"Completed planning for {day}")
        
        return weekly_plan

    def visualize_plan(self, weekly_plan):
        """Create visualizations of the meal plan"""
        # Prepare data for plotting
        nutrition_data = []
        costs = []
        
        for day_plan in weekly_plan:
            nutrition = day_plan['nutrition']
            nutrition_data.append({
                'Day': day_plan['day'],
                'Calories': nutrition['calories'],
                'Protein': nutrition['protein'],
                'Fat': nutrition['fat'],
                'Carbs': nutrition['carbs']
            })
            costs.append(day_plan['total_cost'])
        
        # Create nutrition plot
        df_nutrition = pd.DataFrame(nutrition_data)
        plt.figure(figsize=(15, 8))
        df_nutrition.set_index('Day').plot(kind='bar', rot=45)
        plt.title('Daily Nutritional Values')
        plt.ylabel('Amount')
        plt.tight_layout()
        plt.savefig('nutrition_plot.png')
        plt.close()
        
        # Create cost plot
        plt.figure(figsize=(10, 6))
        plt.bar(self.days, costs)
        plt.title('Daily Meal Costs')
        plt.ylabel('Cost (Rials)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('cost_plot.png')
        plt.close()

    def print_plan(self, weekly_plan):
        """Print formatted meal plan"""
        print("\n" + "="*80)
        print("WEEKLY MEAL PLAN".center(80))
        print("="*80 + "\n")
        
        for day_plan in weekly_plan:
            print(f"\n{day_plan['day'].upper()}")
            print("-" * 60)
            
            if not day_plan['meals']:
                print("No meals planned for this day")
                continue
                
            meals_table = []
            for meal in day_plan['meals']:
                meals_table.append([
                    meal['meal_type'].capitalize(),
                    meal['dish_name'],
                    f"{meal['portion']}g",
                    f"{meal['calories']:.0f} cal",
                    f"{meal['protein']:.1f}g protein",
                    f"{meal['cost']:,.0f} Rials"
                ])
            
            print(tabulate(meals_table, 
                         headers=['Meal', 'Dish', 'Portion', 'Calories', 'Protein', 'Cost'],
                         tablefmt='grid'))
            
            nutrition = day_plan['nutrition']
            print(f"\nDaily Totals:")
            print(f"  Calories: {nutrition['calories']:.0f}")
            print(f"  Protein: {nutrition['protein']:.1f}g")
            print(f"  Fat: {nutrition['fat']:.1f}g")
            print(f"  Carbs: {nutrition['carbs']:.1f}g")
            print(f"  Total Cost: {day_plan['total_cost']:,.0f} Rials")

def main():
    try:
        print("Loading data...")
        df = pd.read_csv('OR_project - Sheet1 (2).csv')
        
        planner = MealPlanner()
        
        print("\nGenerating weekly meal plan...")
        weekly_plan = planner.plan_week(df)
        
        print("\nCreating visualizations...")
        planner.visualize_plan(weekly_plan)
        
        print("\nPrinting meal plan...")
        planner.print_plan(weekly_plan)
        
        print("\nMeal planning complete!")
        print("Check nutrition_plot.png and cost_plot.png for visualizations")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()