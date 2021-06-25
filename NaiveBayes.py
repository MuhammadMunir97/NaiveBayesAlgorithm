import sys

def get_column_rows(data_set, column):
    rows = []
    for row in data_set:
        rows.append (row[column])
    
    return rows

def get_data_set_given_class(data, class_val, class_column_idx):
    return_data = []
    for row in data:
        if int (row[class_column_idx]) == class_val:
            return_data.append (row)

    return return_data

def get_column_data_dict (data, headers):
    columnn_dict = {}
    for idx, header in enumerate (headers):
        column_array = get_column_rows (data, idx)
        columnn_dict[header] = column_array

    return columnn_dict

def build_naive_bayes_model (column_data, headers):
    temp_model = []
    total_values = len (column_data [headers[0]])

    for attribute in headers:
        attibute_arr = column_data [attribute]
        total_zeros = attibute_arr.count ('0')
        probabilty_attribute_zero = round (total_zeros / total_values, 2)
        probabilty_attribute_one = round (1 - probabilty_attribute_zero, 2) # using total probability
        temp_model.append ([probabilty_attribute_zero, probabilty_attribute_one])

    return temp_model

def naive_bayes_model_to_string (naive_bayes_model, class_prob, headers):
    string_to_return = ''
    for class_given, attributes in enumerate (naive_bayes_model):
        string_to_return += "P(class=%s"%(class_given) + ")=%s"%class_prob[class_given] + " "
        for attr_idx, attribute in enumerate (attributes):
            for idx, attribute_val_prob in enumerate (attribute):
                string_to_return += "P(" + headers[attr_idx] + "=%s"%idx + " | %s"%class_given + " )=%s"%attribute_val_prob + " "
        string_to_return += "\n"

    return string_to_return

def classify_row (row, naive_bayes_model, class_probs):
    class_zero_model = naive_bayes_model [0]
    class_one_model = naive_bayes_model [1]

    class_zero_prob = class_probs [0]
    class_one_prob = class_probs [1]
    for idx, attribute_val in enumerate (row):
        class_zero_prob *= class_zero_model [idx] [int (attribute_val)]
        class_one_prob *= class_one_model [idx] [int (attribute_val)]

    return (1 if class_one_prob > class_zero_prob else 0)

def get_model_accuray (data, naive_bayes_model, class_probs, class_val_idx):
    total_vals = len (data)
    correct_classifications = 0
    for row in data:
        expected_classification = int (row [class_val_idx])
        actual_classification = classify_row (row [0: class_val_idx], naive_bayes_model, class_probs)
        if actual_classification == expected_classification:
            correct_classifications += 1

    model_accuracy_percent = (correct_classifications / total_vals) * 100
    return round (model_accuracy_percent, 2)

# ####################################
#
# Running the Application proceduraly
#
# #####################################

if len (sys.argv) != 3:
    raise ValueError("Please provide two arguments")

training_set_file_path = sys.argv [1]
test_set_file_path = sys.argv [2]

training_set = [i.strip().split() for i in open(training_set_file_path).readlines()]
test_set = [i.strip().split() for i in open(test_set_file_path).readlines()]

# index for the class value, last index of the training set
class_val_idx = len (training_set[0]) - 1
headers = training_set [0][0:class_val_idx]

data = training_set [1:]
test_data = test_set [1:]

# the model is an array for array of arrays (3d), the first index will corelate to the class value which is given
# the second index will corelate to the attribute index as presented in the training set, eg. if column 1 is A1, then index 0 -> is A1
# the third index will corelate to attribute value and it will hold the probability  of (A1 | class = 0)  
naive_bayes_model = []

data_set_given_class_zero = get_data_set_given_class (data, 0, class_val_idx)
data_set_given_class_one = get_data_set_given_class (data, 1, class_val_idx)

# Calculate class 0 and 1 probability
total_data_vals = len (data)
total_class_zero_vals = len (data_set_given_class_zero)
class_zero_probability = round ((total_class_zero_vals / total_data_vals), 2)
class_one_probability = round ((1 - class_zero_probability), 2) # using total probability
class_probs = [class_zero_probability, class_one_probability]

class_zero_column_data = get_column_data_dict (data_set_given_class_zero, headers)
class_one_column_data = get_column_data_dict (data_set_given_class_one, headers)

probabilities_given_class_zero = build_naive_bayes_model (class_zero_column_data, headers)
probabilities_given_class_one = build_naive_bayes_model (class_one_column_data, headers)

naive_bayes_model.append(probabilities_given_class_zero)
naive_bayes_model.append(probabilities_given_class_one)

print (naive_bayes_model_to_string (naive_bayes_model, class_probs, headers))

training_set_accuray = get_model_accuray (data, naive_bayes_model, class_probs, class_val_idx)
test_set_accuray = get_model_accuray (test_data, naive_bayes_model, class_probs, class_val_idx)

training_accuracy_to_print = "Accuracy on training set (%s"%len(data) + " instances): %s"% training_set_accuray + "%"
test_accuracy_to_print = "Accuracy on test set (%s"%len(test_data) + " instances): %s"% test_set_accuray + "%"

print (training_accuracy_to_print)
print (test_accuracy_to_print)
