import os
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


#
parser = argparse.ArgumentParser()
parser.add_argument('--path', required = True, type = str)
parser.add_argument('--test_size', default = 0.3, type = float)
opt = parser.parse_args()


#
train_csv_path = os.path.join(opt.path, "train.csv")
test_csv_path = os.path.join(opt.path, "test.csv")
classes_path = os.path.join(opt.path, "classes.txt")


#
category_names = [category
                 for category in os.listdir(opt.path)
                 if os.path.isdir(os.path.join(opt.path, category))]


#
X, Y = [], []
for category_name in category_names:

        class_names = [class_name
                       for class_name in os.listdir(os.path.join(opt.path, category_name))
                       if os.path.isdir(os.path.join(opt.path, category_name, class_name))]
        
        for class_name in class_names:

            imgs_paths = [os.path.join(category_name, class_name, img_name) 
                          for img_name in os.listdir(os.path.join(opt.path, category_name, class_name))
                          if img_name.endswith("jpg") or img_name.endswith("jpeg")]
            
            for img_path in imgs_paths:
                X.append(img_path)
                Y.append(class_name)

df = pd.DataFrame(data = {'image_name': X, 'category_name': Y} )

X_train, X_test, y_train, y_test = train_test_split(df['image_name'], df['category_name'], test_size = opt.test_size, random_state = 42, shuffle = True)

d = {'image_name': X_train, 'category_name': y_train}
pd.DataFrame(data = d).to_csv(train_csv_path, index = False)

d = {'image_name': X_test, 'category_name': y_test}
pd.DataFrame(data = d).to_csv(test_csv_path, index = False)

with open(classes_path, 'w') as output:
    for row in sorted(df['category_name'].unique()): 
        output.write(str(row) + '\n')
        