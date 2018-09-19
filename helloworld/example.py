
from keras.models import Sequential
from keras.layers import Dense
import numpy

filepath="./pima-indians-diabetes.csv"

# 5. Number of Instances: 768
# 
# 6. Number of Attributes: 8 plus class 
# 
# 7. For Each Attribute: (all numeric-valued)
#    1. Number of times pregnant
#    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#    3. Diastolic blood pressure (mm Hg)
#    4. Triceps skin fold thickness (mm)
#    5. 2-Hour serum insulin (mu U/ml)
#    6. Body mass index (weight in kg/(height in m)^2)
#    7. Diabetes pedigree function
#    8. Age (years)
#    9. Class variable (0 or 1)
# 
# 8. Missing Attribute Values: Yes
# 
# 9. Class Distribution: (class value 1 is interpreted as "tested positive for
#    diabetes")
# 
#    Class Value  Number of instances
#    0            500
#    1            268



# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt(filepath, delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10) #verbose=0 to remove progres vars

# evaluate the model
scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# Make Predictions
print("#"*25)
print("Time to predict!")
print("#"*25)
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(len(rounded), rounded)
print(len(Y), Y)
notequal = 0
for i in range(len(rounded)):
    if rounded[i] != Y[i]:
        notequal += 1
        print("#"+str(i), Y[i]," (original) != (predict) ", rounded[i] )

print("%.2f%%, %.2f%%" % (scores[1]*100, 100-float(notequal)/len(rounded)*100))
