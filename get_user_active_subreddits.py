
import requests

def get_user_posts(author_name, num_posts, sub_or_com, after_value):
    try:
        url = f'https://api.pushshift.io/reddit/search/{sub_or_com}/?author={author_name}&sort=desc&before=12m&size={num_posts}'
        #print(url)
        data = requests.get(url, headers={'User-Agent': 'Ubuntu: requests (by /u/Rise_Above_The_Flame)'})

        content = data.json()['data']
        return content

    except:
        print("Error getting data from Pushshift API")
        exit()

# posts --> comments or posts to be added to the dict
# posts_string --> either 'comment' or 'submission' depending on which we're processing
def get_subreddits(posts, posts_string, dict_name):
    for post in posts:
        subreddit = post['subreddit']
        if subreddit in dict_name:
            if posts_string in dict_name[subreddit]:
                dict_name[subreddit][posts_string] += 1
            else:
                dict_name[subreddit][posts_string] = 1
        else:
            dict_name[subreddit] = dict()
            dict_name[subreddit][posts_string] = 1
    return dict_name

def main():
    user = 'adamwer'
    posts = get_user_posts(user, 5, 'submission', 0)
    comments = get_user_posts(user, 5, 'comment', 0)
    user_stats = dict()
    updated_dict = get_subreddits(posts, 'submissions', user_stats)
    subreddit_frequency = get_subreddits(comments, 'comments', updated_dict)

    print(subreddit_frequency)

if __name__ == '__main__':
    main()
