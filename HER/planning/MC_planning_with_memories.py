import numpy as np 
import math

class Node:
	def __init__(self, state, reward =0., value=0., skill_num = -1, params = None, parent = None, height=-1, prob = 0.):
		self.state = state
		self.child = None
		self.reward = reward
		self.value = value
		self.skill_num = skill_num
		self.params = params
		self.parent = parent
		self.height = height
		self.prob = prob

class Planning_with_memories:
	"""docstring for Planning_with_memories"""
	def __init__(self, skillset, env, num_samples=2, max_height =3, value_scale=0., reward_scale=1.):
		self.skillset = skillset
		self.env = env
		self.num_samples = num_samples
		self.max_height = max_height

		# value_scale*(sum of values) + reward_scale*(sum of rewards)
		self.value_scale = value_scale
		self.reward_scale = reward_scale
		
	def create_plan(self, state, height = None):
		
		info = dict()
		
		openlist = []
		leaflist = []

		if height is None:
			# root call
			height = self.max_height
			openlist.append(Node(state=state, height=height, prob=1.))

		while(openlist):
			curr_node = openlist.pop(0)

			# reached a leaf node
			if(curr_node.height==0):
				leaflist.append(curr_node)
				continue

			# create child node
			# sample skills
			sampled_skills = np.random.choice(self.skillset.len, size = self.num_samples)
			# print(sampled_skills)
			for node_id, sampled_skill_num in enumerate(sampled_skills):
				sampled_params = np.random.uniform(low=-1,high=1, size=self.skillset.num_skill_params(sampled_skill_num))

				# print("skill:%d, goal"%sampled_skill_num, sampled_params)
				
				critic_value = self.skillset.get_critic_value(primitive_id=sampled_skill_num, obs = state, primitive_params = sampled_params)
				next_state = self.skillset.get_terminal_state_from_memory(primitive_id=sampled_skill_num,obs = state, primitive_params = sampled_params)
				prob = self.skillset.get_prob_skill_success(primitive_id=sampled_skill_num,obs = state, primitive_params = sampled_params)
				
				# print("prob:%.4f"%prob)
				if curr_node.height == 1:
					# creating a leaf node
					child_reward = self.env.calc_reward(next_state)
				else:
					child_reward = 0.

				new_node = Node(state = next_state, 
								value= critic_value + curr_node.value, 
								reward = child_reward + curr_node.reward, 
								skill_num = sampled_skill_num, params = sampled_params, 
								parent = curr_node, height = curr_node.height -1,
								prob = curr_node.prob*prob)

				# no need to sort as we want all the paths to be explored
				openlist.append(new_node)

		max_utility = -math.inf
		max_utility_node_id = -1
		# get the child with max utility
		for node_id, node in enumerate(leaflist):
			#print("value:%.4f, reward:%.4f"%(node.value, node.reward))
			# if(node.skill_num>0):
			# 	print("obj loc:%s"%(str(node.state[3:6])))
			# else:
			# 	print("Reacher")
			node_utility = (self.value_scale*node.value + self.reward_scale*node.reward)

			expected_node_utility = node.prob*node_utility

			if(node_utility > max_utility):
				max_utility = node_utility
				max_utility_node_id = node_id


		# get the root node and skills
		curr_node = leaflist[node_id]
		while(curr_node.parent is not None):
			parent_node = curr_node.parent
			parent_node.child = curr_node
			curr_node = parent_node

		chosen_skill = curr_node.child.skill_num
		chosen_skill_params = curr_node.child.params

		# create paction and return
		#print("suggested skill id:%d, utility:%.4f,goal:"%(chosen_skill, max_utility), chosen_skill_params)
		paction = np.zeros((self.skillset.len + self.skillset.num_params, ))
		paction[chosen_skill] = 1.0
		starting_idx = self.skillset.params_start_idx[chosen_skill]
		ending_idx = starting_idx + chosen_skill_params.size

		paction[ starting_idx: ending_idx] = chosen_skill_params
		
		info['next_state'] = (curr_node.child.state)
		info['prob'] = curr_node.prob
		print("prob:%.4f, value:%.4f, goal"%(curr_node.prob, curr_node.value), chosen_skill_params)
		
		return paction.copy(), info




		





		
