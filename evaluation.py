import os
import pandas as pd
from sklearn.metrics import classification_report
from visualization import plot_confusion_matrix

def evaluate_predictions(df, labels_df, class_labels, st):
    # Normalize filenames before merging
    df['Filename'] = df['Filename'].apply(lambda x: os.path.basename(str(x).strip().lower()))
    labels_df['Filename'] = labels_df['Filename'].apply(lambda x: os.path.basename(str(x).strip().lower()))
    merged_df = pd.merge(df, labels_df, on='Filename')
    st.dataframe(merged_df)
    y_true = merged_df['class']
    y_pred = merged_df['Prediction']
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df)
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred, labels=class_labels, st=st)
