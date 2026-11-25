import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn import metrics
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve


def report_regression_metrics(y_train, y_train_pred, y_test, y_test_pred, model):
    from sklearn.metrics import mean_squared_error, r2_score

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print('Train MSE of', model, 'is ', round(train_mse, 5), '\n')
    print('Test MSE of', model, 'is ', round(test_mse, 5), '\n\n')
    print('Train R2 of', model, 'is ', round(train_r2, 5), '\n')
    print('Test R2 of', model, 'is ', round(test_r2, 5), '\n\n')


def report_classification_metrics(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    classifier,
    y_test_score=None,
):
    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)

    print('Train accuracy score of', classifier, 'is ', round(train_score, 5), '\n')
    print('Test accuracy score of', classifier, 'is ', round(test_score, 5), '\n\n')

    train_c_matrix = confusion_matrix(y_train, y_train_pred)
    plt.figure(figsize=(4, 4))
    plt.title('Confusion Matrix for the Train Subset')
    sns.heatmap(train_c_matrix, annot=True, fmt=".1f", linewidths=.5, square=True, cmap='Pastel1')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Train Accuracy: {0}'.format(round(train_score, 4)), size=15)
    plt.show()

    print('Classification Report for the Train Subset: \n\n', classification_report(y_train, y_train_pred))

    test_c_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(4, 4))
    plt.title('Confusion Matrix for the Test Subset')
    sns.heatmap(test_c_matrix, annot=True, fmt=".1f", linewidths=.5, square=True, cmap='Pastel1')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Test Accuracy: {0}'.format(round(test_score, 4)), size=15)
    plt.show()

    print('Classification Report for the Test Subset: \n\n', classification_report(y_test, y_test_pred))

    fpr, tpr, _ = roc_curve(y_test, y_test_pred if y_test_score is None else y_test_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.title('ROC Curve for the Test Subset')
    plt.plot(fpr, tpr, color='red', label=classifier + ' (AUC: %.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.title('ROC Curve for the Test Subset')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.show()


def plot_decision_regions_for_data(clf, X, y, feature_names, title="Decision Regions for Data"):
    plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.gcf().set_size_inches(7, 5)
    plt.show()
