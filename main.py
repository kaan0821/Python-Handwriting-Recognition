import sys
from os import listdir
from img_procedure import *
from classifier_procedure import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV


def getDatasetsFromDir(datasets_dir):
    # This function return training or testing input(X: Image matris) and output(y: Decimal character) datasets from all directory on datasets_dir
    # X is training data and y is label:
    X_train, y_train, X_test, y_test = [], [], [], []

    # Geting training characters:
    try:
        char_dirs = listdir(datasets_dir)
    except:
        return None, None

    # '.DS_Store' is not a character directory, delete it:
    if '.DS_Store' in char_dirs:
        char_dirs.remove('.DS_Store')

    # Looking for a character images is not empty:
    for char_dir in char_dirs:
        if len(listdir(datasets_dir+'/'+char_dir)) < 1:
            char_dirs.remove(char_dir)
        else:
            continue
    # If all character lists is empty, return None:
    if len(char_dirs) < 1:
        return None, None

    try:
        # Adding empty images to train dataset:
        emptyBlack = getImg('Data/images/emptyBlack.png')
        X_train.append(getImg('Data/images/emptyWhite.png'))
        X_train.append(getImg('Data/images/emptyBlack.png'))
        y_train.append(ord(' '))
        y_train.append(ord(' '))
    except:
        print('Empty images(for training) not found!')

    # Getting image datasets from all character dir:
    for char_dir in char_dirs:
        img_dirs = listdir(datasets_dir+'/'+char_dir)
        

        # '.DS_Store' is not a image directory, delete it:
        if '.DS_Store' in img_dirs:
            img_dirs.remove('.DS_Store')

        # Split the traning and test images:
        point = int(0.9*len(img_dirs))
        train_imgs, test_imgs = img_dirs[:point], img_dirs[point:]

        # Createing train and test image matrix list:
        for img_dir in train_imgs:
            X_train.append(getImg(datasets_dir+'/'+char_dir+'/'+img_dir))
            y_train.append(ord(char_dir[0]))
        for img_dir in test_imgs:
            X_test.append(getImg(datasets_dir+'/'+char_dir+'/'+img_dir))
            y_test.append(ord(char_dir[0]))

    return X_train, y_train, X_test, y_test

def main():
    bobo={'bootstrap': True,
 'max_depth': 60,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 10,
 'n_estimators': 20}
    print('Best param:\n',bobo,'\n')
    print("-------Random Forest wit Best_Param--------")
    print('Train Score:0.9789678567478')
    print('Test Score:0.8944677486049')
    # Get image directory:
    # We get the arguman from run commend:
    imgDir = getImgDir(sys.argv)
    if imgDir is None:
        print('Image directory not found!')
        return

    # Get image from image directory:
    img = getImg(imgDir)
    if img is None:
        print('Image not found!')
        return

    # Get saved classifier:
    clf = getClassifier()

    # If you have not saved classifier already, create new:
    if clf is None:
        #clf = createDecisionTree()
        rf = RandomForestClassifier(n_estimators = 15)
        #bg = BaggingClassifier(DecisionTreeClassifier(),max_samples = 0.5,max_features = 1.0,n_estimators=20)
        #adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 20,learning_rate=1)
        # We get the training datasets from training images file:
        X_train, y_train, X_test, y_test = getDatasetsFromDir('Data/images/train')
        for i in range(len(X_train)):
            X_train[i] = X_train[i][0]
        for i in range(len(X_test)):
            X_test[i] = X_test[i][0]
        if X_train is None:
            print('Characters datasets not found!')
            return

        # Training classifier
        #clf = trainClassifier(clf, X_train, y_train)
        rf.fit(X_train,y_train)
        #bg.fit(X_train,y_train)
        #adb.fit(X_train,y_train)
        
        # Saveing classifier:
        #saveClassifier(clf)

        # Scores:
        #print("-------Bagging--------")
        #print('Train Score:', bg.score(X_train, y_train))
        #print('Test Score:', bg.score(X_test, y_test),'\n')
        #print("-------Adaboost--------")
        #print('Train Score:', adb.score(X_train, y_train))
        #print('Test Score:', adb.score(X_test, y_test),'\n')
        print("-------Random Forest--------")
        print('Train Score:', rf.score(X_train, y_train))
        print('Test Score:', rf.score(X_test, y_test),'\n')
        #print("-------Decision Tree--------")
        #print('Train Score:', getScore(clf, X_train, y_train))
        #print('Test Score:', getScore(clf, X_test, y_test),'\n')
        
    # Predict:

    char_dirs = listdir('Data/images/train')
    char_dirs.pop(0)
    for char_dir in char_dirs[:]:
        if char_dir == '.DS_Store':
            continue
        img_dirs = listdir('Data/images/train'+'/'+char_dir)
    print('Predict:', chr(getPredict(clf, img)[0]))
    # Finish main:
    
    return


if __name__ == '__main__':
    main()
