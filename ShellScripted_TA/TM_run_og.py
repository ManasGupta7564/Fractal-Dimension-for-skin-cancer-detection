import ctypes
import numpy as np
import os
import csv
import argparse
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

def train_model(X_train,Y_train,features,num_classes,s,num_epochs):
    lib_path = os.path.join(os.path.dirname(__file__), "tsetlin.so")
    lib = ctypes.CDLL(lib_path)
    X_tr=np.load(X_train,allow_pickle=True)
    Y_tr=np.load(Y_train,allow_pickle=True)
    lib.CreateTsetlinMachine.restype=ctypes.c_void_p
    lib.CreateTsetlinMachine.argtypes=[]

    #tm_update
    lib.tm_update.restype=None
    lib.tm_update.argtypes= [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_float
    ]
    #tm_score
    lib.tm_score.restype=ctypes.c_int
    lib.tm_score.argtypes=[
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int)
    ]
    lib.tm_show_value.restype = ctypes.c_uint
    lib.tm_show_value.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.tm_get_state.restype = ctypes.c_int
    lib.tm_get_state.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    tmachines = [lib.CreateTsetlinMachine() for _ in range(num_classes)]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for Xi_row, true_label in zip(X_tr, Y_tr):
            # Convert feature vector to ctypes array
            Xi_ctypes = (ctypes.c_int * features)(*Xi_row.tolist())
            # Update each class-specific TM
            for cls in range(num_classes):
                target = 1 if cls == true_label else 0
                lib.tm_update(tmachines[cls], Xi_ctypes, target, ctypes.c_float(s))
        print("Epoch complete.")

    return tmachines, lib


def save_model(lib,tmachines,feat,clause,save_dir="/Model_Saved"):
    lib.tm_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tm_save.restype = ctypes.c_int
    os.makedirs(save_dir, exist_ok=True)

    for cls, tm in enumerate(tmachines):
        fname = os.path.join(save_dir, f"tm_class{cls}.bin").encode('ascii')
        ret = lib.tm_save(tm, fname)
        if ret != 0:
            raise RuntimeError(f"tm_save failed for class {cls} (err {ret})")

    # Also save the TA_States
    lib.tm_get_state.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.tm_get_state.restype = ctypes.c_int
    os.makedirs("TA_STATE", exist_ok=True)
    path = f"TA_STATE/TA_{feat}_{clause}.txt"
    with open(path,"w") as f:
        for cls, tm in enumerate(tmachines):
            for i in range(clause):
                for j in range(feat):
                    pos=lib.tm_get_state(tm,i,j,0)
                    neg=lib.tm_get_state(tm,i,j,1)
                    f.write(f"{pos} {neg} ")
        f.seek(f.tell() - 1)
        f.truncate()



def predict(lib, tmachines, X_test, features, num_classes, cls_sm_dir="./Class_Sums"):
    os.makedirs(cls_sm_dir, exist_ok=True)
    results = []
    predicted_labels = []

    for i in range(X_test.shape[0]):
        Xi_test = (ctypes.c_int * features)(*X_test[i].tolist())
        class_sums = [lib.tm_score(tm, Xi_test) for tm in tmachines]
        predicted_label = int(np.argmax(class_sums))
        predicted_labels.append(predicted_label)
        results.append([i] + class_sums + [predicted_label])

    # Save to CSV
    csv_filename = os.path.join(cls_sm_dir, f"features_{features}.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["sample_index"] + [f"Class_{cls}" for cls in range(num_classes)] + ["predicted_label"]
        writer.writerow(header)
        writer.writerows(results)

    print(f"Predictions saved to {csv_filename}")
    return np.array(predicted_labels), results


def evaluate_model(predicted_labels, true_labels):
    acc = np.mean(predicted_labels == true_labels)
    print(f"\nAccuracy: {acc * 100:.2f}%")

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    print(f"Precision (Cancer=1): {precision:.4f}")
    print(f"Recall (Cancer=1): {recall:.4f}")
    print(f"F1-Score (Cancer=1): {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

    print("\nFull Classification Report:")
    print(classification_report(true_labels, predicted_labels, zero_division=0))

    return acc

#Our state variables are : X_train,X_test,Y_train,Y_test,num_classes,s,num_epochs
def main():
    parser = argparse.ArgumentParser(description="Train and Test Tsetlin Machine Model")
    parser.add_argument("--X_train", type=str, required=True, help="Path to X_train .npy file")
    parser.add_argument("--Y_train", type=str, required=True, help="Path to Y_train .npy file")
    parser.add_argument("--X_test", type=str, required=True, help="Path to X_test .npy file")
    parser.add_argument("--Y_test", type=str, required=True, help="Path to Y_test .npy file")
    parser.add_argument("--features", type=int, required=True, help="Number of features")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--s", type=float, required=True, help="Specificity parameter")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--clauses", type=int, required=True, help="Number of Clauses")
    parser.add_argument("--save_dir", type=str, default="./Model_Saved", help="Directory to save model")

    args = parser.parse_args()

    # Training
    tmachines, lib = train_model(args.X_train, args.Y_train, args.features, args.num_classes, args.s, args.num_epochs)

    # Save model
    save_model(lib, tmachines, args.features, args.clauses, args.save_dir)

    # Load test data
    X_te = np.load(args.X_test,allow_pickle=True)
    Y_te = np.load(args.Y_test,allow_pickle=True)

    # Predict and Accuracy
    predicted_labels, res = predict(lib, tmachines, X_te, args.features, args.num_classes)
    evaluate_model(predicted_labels, Y_te)

if __name__ == "__main__":
    main()



