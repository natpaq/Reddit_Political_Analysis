import json
import matplotlib.pyplot as plt


def sort_submissions(data):
    sorted_data = sorted(data.items(), key=lambda x: x[1]['submissions'], reverse=True)
    x = []
    for k, v in sorted_data:
        x.append(tuple((k, v['submissions'])))
    return x


def sort_comments(data):
    sorted_data = sorted(data.items(), key=lambda x: x[1]['comments'], reverse=True)
    x = {}
    for k, v in sorted_data:
        x[k] = v['comments']
    return x


def sort_by_activity(data):
    sorted_data = sorted(data.items(), key=lambda x: (x[1]['submissions'], x[1]['comments']), reverse=True)
    x = {}
    for k, v in sorted_data:
        x[k] = v
    return x


def plot_top(sorted_data, title, y_label, x_label):
    top = sorted_data[:10]
    keys = [i[0] for i in top]
    vals = [i[1] for i in top]
    fig, ax = plt.subplots()
    ax.bar(keys, vals)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()

def do_it():
    # First take care of Biden
    with open('./user_data/user_data/biden_subreddit_freq.json') as json_file:
        data = json.load(json_file)
    sorted_data = sort_submissions(data)
    plot_top(sorted_data, 'Biden User Subreddit Frequency of Usage', 'Submissions', 'Subreddit')

    # Now take care of Trump

if __name__ == '__main__':
    do_it()