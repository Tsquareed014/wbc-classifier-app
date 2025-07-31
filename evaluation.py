import os
import pandas as pd
from sklearn.metrics import classification_report
from visualization import plot_confusion_matrix

def evaluate_predictions(df, labels_df, class_labels, st):
    # Normalize filenames before merging
    df['Filename'] = df['Filename'].apply(lambda x: os.path.basename(str(x).strip().lower()))
    labels_df['Filename'] = labels_df['Filename'].apply(lambda x: os.path.basename(str(x).strip().lower()))
    merged_df = pd.merge(df, labels_df, on='Filename', how='inner')
    
    # Display merged data for debugging
    st.dataframe(merged_df)
    
    # Check and correct column names
    true_label_col = next((col for col in ['Class', 'class'] if col in merged_df.columns), None)
    if true_label_col is None:
        st.error("Error: True label column ('Class' or 'class') not found in merged data.")
        return
    
    if 'Prediction' not in merged_df.columns:
        st.error("Error: Prediction column not found in merged data.")
        return
    
    y_true = merged_df[true_label_col]
    y_pred = merged_df['Prediction']
    
    # Generate and display classification report
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df)
    except ValueError as e:
        st.error(f"Error generating classification report: {str(e)}")
    
    # Generate and display confusion matrix
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred, labels=class_labels, st=st)
