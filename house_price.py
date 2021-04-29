import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
        
house = pd.read_csv("train.csv")
house_test = pd.read_csv("test_csv")
house.describe()
house.info()
house.head()
house.isnull().sum()

df = house.drop(['Id','Street','Alley','MasVnrType', 'LandContour' , 'Utilities' , 'Condition2' , 'PoolQC' , 'Fence' , 'MiscFeature' , 'FireplaceQu' ],axis=1,inplace=True)

for i in range(0,70):
    if (house.iloc[:,i].isnull().sum()>0):
        print(house.columns[i] , house.iloc[:,i].isnull().sum())
        
house['LotFrontage'] = house['LotFrontage'].fillna(house['LotFrontage'].mean())
house['MasVnrArea'] = house['MasVnrArea'].fillna(house['MasVnrArea'].mean())
house['BsmtQual'] = house['BsmtQual'].fillna(house['BsmtQual'].mode()[0])
house['BsmtCond'] = house['BsmtCond'].fillna(house['BsmtCond'].mode()[0])
house['BsmtExposure'] = house['BsmtExposure'].fillna(house['BsmtExposure'].mode()[0])
house['BsmtFinType1'] = house['BsmtFinType1'].fillna(house['BsmtFinType1'].mode()[0])
house['BsmtFinType2'] = house['BsmtFinType2'].fillna(house['BsmtFinType2'].mode()[0])
house['Electrical'] = house['Electrical'].fillna(house['Electrical'].mode()[0])
house['GarageType'] = house['GarageType'].fillna(house['GarageType'].mode()[0])
house['GarageFinish'] = house['GarageFinish'].fillna(house['GarageFinish'].mode()[0])
house['GarageQual'] = house['GarageQual'].fillna(house['GarageQual'].mode()[0])
house['GarageCond'] = house['GarageCond'].fillna(house['GarageCond'].mode()[0])
house['GarageYrBlt'] = house['GarageYrBlt'].fillna(house['GarageYrBlt'].mode()[0])


house['MSSubClassLog'] = np.log(house['MSSubClass'])
house['LotFrontageLog'] = np.log(house['LotFrontage'])
house['LotAreaLog'] = np.log(house['LotArea'])
house['OverallQuallog'] = np.log(house['OverallQual'])
house['OverallCondlog'] = np.log(house['OverallCond'])
house['YearBuiltlog'] = np.log(house['YearBuilt'])
house['YearRemodAddlog'] = np.log(house['YearRemodAdd'])
house['MasVnrArealog'] = np.log(house['MasVnrArea'])
house['BsmtFinSF1log'] = np.log(house['BsmtFinSF1'])
house['BsmtFinSF2log'] = np.log(house['BsmtFinSF2'])
house['BsmtUnfSFlog'] = np.log(house['BsmtUnfSF'])
house['TotalBsmtSFlog'] = np.log(house['TotalBsmtSF'])
house['1stFlrSFlog'] = np.log(house['1stFlrSF'])
house['2ndFlrSFlog'] = np.log(house['2ndFlrSF'])
house['LowQualFinSFlog'] = np.log(house['LowQualFinSF'])
house['GrLivArealog'] = np.log(house['GrLivArea'])
house['BsmtFullBathlog'] = np.log(house['BsmtFullBath'])
house['BsmtHalfBathlog'] = np.log(house['BsmtHalfBath'])
house['FullBathlog'] = np.log(house['FullBath'])
house['HalfBathlog'] = np.log(house['HalfBath'])
house['BedroomAbvGrlog'] = np.log(house['BedroomAbvGr'])
house['KitchenAbvGrlog'] = np.log(house['KitchenAbvGr'])
house['TotRmsAbvGrdlog'] = np.log(house['TotRmsAbvGrd'])
house['Fireplaceslog'] = np.log(house['Fireplaces'])
house['GarageYrBltlog'] = np.log(house['GarageYrBlt'])
house['GarageCarslog'] = np.log(house['GarageCars'])
house['GarageArealog'] = np.log(house['GarageArea'])
house['WoodDeckSFlog'] = np.log(house['WoodDeckSF'])
house['OpenPorchSFlog'] = np.log(house['OpenPorchSF'])
house['EnclosedPorchlog'] = np.log(house['EnclosedPorch'])
house['3SsnPorchlog'] = np.log(house['3SsnPorch'])
house['ScreenPorchlog'] = np.log(house['ScreenPorch'])
house['PoolArealog'] = np.log(house['PoolArea'])
house['MiscVallog'] = np.log(house['MiscVal'])
house['MoSoldlog'] = np.log(house['MoSold'])
house['YrSoldlog'] = np.log(house['YrSold'])
#house['SalePricelog'] = np.log(house['SalePrice'])


#correlation matrix
corr = house.corr()
sns.heatmap(corr,annot= True ,cmap = "BuPu" )

dff = house.drop(['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold'],axis = 1 ,inplace = True)


cols = ['MSZoning','LotShape','LotConfig','LandSlope','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']
le=LabelEncoder()
for col in cols:
    house[col]=le.fit_transform(house[col])
    
X=house.drop(columns='SalePrice',axis=1)
y=house['SalePrice']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


from xgboost import XGBRegressor
rgr = XGBRegressor(n_estimators=500, learning_rate= 0.05)
rgr.fit(X_train, y_train)
y_pred = rgr.predict(X_test)


    

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
