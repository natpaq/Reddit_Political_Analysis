import json
import matplotlib.pyplot as plt
import numpy as np


def sort_submissions(data):
    sorted_data = sorted(data.items(), key=lambda x: x[1]['submissions'], reverse=True)
    x = []
    for k, v in sorted_data:
        x.append(tuple((k, v['submissions'])))
    return x


def sort_comments(data):
    sorted_data = sorted(data.items(), key=lambda x: x[1]['comments'], reverse=True)
    x = []
    for k, v in sorted_data:
        x.append(tuple((k, v['comments'])))
    return x


def sort_by_activity(data):
    sorted_data = sorted(data.items(), key=lambda x: (x[1]['submissions'], x[1]['comments']), reverse=True)
    x = {}
    for k, v in sorted_data:
        x[k] = v
    return x


def sort_by_interaction(data):
    sorted_data = sorted(data.items(), key=lambda x: x[1]['submissions'] + x[1]['comments'], reverse=True)
    x = []
    for k, v in sorted_data:
        x.append(tuple((k, v['comments'] + v['submissions'])))
    return x


def plot_top(submissions, comments, title):
    fig, axs = plt.subplots(2)
    # submissions
    top = submissions[:10]
    keys = [i[0] for i in top]
    vals = [i[1] for i in top]
    axs[0].bar(keys, vals)
    axs[0].set_ylabel('Submissions')
    # comments
    top = comments[:10]
    keys = [i[0] for i in top]
    vals = [i[1] for i in top]
    axs[1].bar(keys, vals, color=['r'])
    axs[1].set_ylabel('Comments')
    axs[1].set_xlabel('Subreddit')
    # plot
    plt.setp(axs[0].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(axs[1].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('./user_data/figs/' + title + '.png')
    # plt.show()


def plot_interaction(interactions, submissions, comments, title):
    top = interactions[:10]
    interaction_labels = [i[0] for i in top]

    subs = []
    sub_dic = dict(submissions)
    for i in interaction_labels:
        if sub_dic[i]:
            subs.append(sub_dic[i])

    coms = []
    com_dic = dict(comments)
    for i in interaction_labels:
        if com_dic[i]:
            coms.append(com_dic[i])

    width = 0.35  # the width of the bars
    x = np.arange(len(interaction_labels))  # the label locations

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, subs, width, label='Submissions')
    rects2 = ax.bar(x + width/2, coms, width, color=['r'], label='Comments')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Interaction')
    ax.set_xlabel('Subreddit')
    ax.set_xticks(x)
    ax.set_xticklabels(interaction_labels)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig('./user_data/figs/' + title + '.png')
    plt.show()


def do_it():
    # First take care of Biden
    with open('./user_data/biden_subreddit_freq.json') as json_file:
        data = json.load(json_file)
    submissions = sort_submissions(data)
    comments = sort_comments(data)
    interactions = sort_by_interaction(data)
    plot_interaction(interactions, submissions, comments, 'Non Political Subreddit Frequency of Usage (Biden)')

    with open('./user_data/biden_subreddit_freq_global.json') as json_file:
        data = json.load(json_file)
    submissions = sort_submissions(data)
    comments = sort_comments(data)
    plot_top(submissions, comments, 'Subreddit Frequency of Usage (Biden)')

    # Now take care of Trump
    with open('./user_data/trump_subreddit_freq.json') as json_file:
        data = json.load(json_file)
    submissions = sort_submissions(data)
    comments = sort_comments(data)
    interactions = sort_by_interaction(data)
    plot_interaction(interactions, submissions, comments, 'Non Political Subreddit Frequency of Usage (Trump)')

    with open('./user_data/trump_subreddit_freq_global.json') as json_file:
        data = json.load(json_file)
    submissions = sort_submissions(data)
    comments = sort_comments(data)
    plot_top(submissions, comments, 'Subreddit Frequency of Usage (Trump)')

if __name__ == '__main__':
    do_it()
