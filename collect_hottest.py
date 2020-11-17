import argparse
import json
import requests

# Will get the input number of posts from the input-specified subreddit
def get_posts(subreddit_name, num_posts, initial, after_value):
# Subreddit_name --> Name of subreddit from which to get posts (expected input form is '/r/politics')
# num_posts --> number of posts to get from subreddit
# initial is 1 if it's the first run or 0 if we want to extract more than 100 posts
# after_value is the identifier of the last previous reddit post that was extracted in previous run
	try:
		if initial == 1:
			data = requests.get(f'http://api.reddit.com{subreddit_name}/hot?limit={num_posts}',
				headers={'User-Agent': 'Ubuntu: requests (by /u/Rise_Above_The_Flame)'})
		else:
			data = requests.get(f'http://api.reddit.com{subreddit_name}/hot?limit={num_posts}&after={after_value}',
				headers={'User-Agent': 'Ubuntu: requests (by /u/Rise_Above_The_Flame)'})

		content = data.json()['data']

		return (content['children'], content['after'])
	except:
		print("Error getting data from Reddit API")
		exit()

# Write to JSON file
def write_to_JSON(posts, output_file):
# posts --> posts returned from Reddit api
# output_file --> file object
		# Write each of these posts to output file on a new line
		for post in posts:
			# Ignore stickied posts
			if post['data']['stickied'] == False:
				json.dump(post['data'], output_file)
				output_file.write('\n')


# Write the posts to specified JSON file
def get_write_posts_to_file(subreddits, post_sample, output_file_path):
# Subreddits --> Subreddit to extract posts from
# Post_sample --> The number of posts to extract from each subreddit
# Output_file_path --> path and name of file where posts should be stored
	with open(output_file_path, "w") as outfile:

		# Account for the fact that 100 is the max number of posts that can be extracted at a time
		if post_sample > 100:
			post_sample -= 100
			num_posts = 100
		# We can extract all posts in one sampling
		else:
			num_posts = post_sample
			post_sample = 0

		# Get first number of posts from each subreddit
		for subreddit in subreddits:
			# Get first batch of posts
			posts, after_value = get_posts(subreddit, num_posts, 1, None)
			write_to_JSON(posts, outfile)

			# Check if we need to extract more posts
			while post_sample != 0:
				if post_sample > 100:
					post_sample -= 100
					num_posts = 100
				else:
					num_posts = post_sample
					post_sample = 0

				posts, after_value = get_posts(subreddit, num_posts, 0, after_value)
				write_to_JSON(posts, outfile)



def main():
    # Specify arguments to be expected
    parser = argparse.ArgumentParser()
    # Require an argument specifying path and file name of output file
    parser.add_argument('-o', help='The path and filename specifying the output file that posts will be written to', required=True)
    # Require an argument specifying the subreddit from which posts should be extracted
    parser.add_argument('subreddit', help='The subreddit from which posts should be extracted from, written as: /r/subreddit_name')
    args = parser.parse_args()
    output_file = args.o
    subreddit = [args.subreddit]

    # Specify number of posts to extract from each subreddit
    num_posts = 500

	# Get and write posts to output JSON file
    get_write_posts_to_file(subreddit, num_posts, output_file)



if __name__ == '__main__':
	   main()
