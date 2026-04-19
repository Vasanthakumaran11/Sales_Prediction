import os
import glob

algos_dir = r"c:\Users\Raja\Desktop\Sales_Prediction\RetailForecasting\algorithms"

files = glob.glob(os.path.join(algos_dir, "*.py"))

old_split = """self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )"""

new_split = """self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )"""

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    
    if old_split in content:
        content = content.replace(old_split, new_split)
        with open(f, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Updated shuffle=False in {os.path.basename(f)}")
