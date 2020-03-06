import pygraphviz as pgv


def get_most_concrete_root_node(ctree):
    curr_node = ctree.root
    while len(curr_node.covered_k_idxs) == len(curr_node.children[0].covered_k_idxs):
        curr_node = curr_node.children[0]
    return curr_node


def get_constraint_in_string(node):
    string_form = ''
    string_form = str(node.constraint.var_types) + '\n'
    for p in node.constraint.value:
        string_form += str(p) + '\n'
    string_form += str(node.covered_k_idxs)
    return string_form


def get_constraint_diff(parent, child):
    pconst = parent.constraint
    cconst = child.constraint
    diff = []
    c_var_types = [p for p in cconst.var_types.iteritems()]
    p_var_types = [p for p in pconst.var_types.iteritems()]
    for vc in c_var_types:
        if vc not in p_var_types:
            diff.append(vc)

    for pc in cconst.value:
        if pc not in pconst.value:
            diff.append(pc)
    return str(diff)


def get_node_info_in_string(node):
    Q=''
    N = ''
    parent_action = ''
    sum_rewards_history=''
    for key, value in zip(node.Q.keys(), node.Q.values()):
        if key[0].find('pick') !=-1 :
            try:
                Q += key[0]+' (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f): %.2f '%(key[1], key[2], key[3], key[4], key[5], key[6], value)
            except:
                Q += 'None'
        else:
            Q +=  key[0]+' (%.2f,%.2f,%.2f):%.2f ' % (key[1], key[2], key[3], value)

    """
    for key, values in zip(node.sum_rewards_history.keys(), node.sum_rewards_history.values()):
        if len(key) == 2:
            sum_rewards_history += '(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f): ' % (key[0][0], key[0][1], key[0][2],
                                                                          key[1][0], key[1][1], key[1][2]) + str(values)
        else:
            sum_rewards_history += '(%.2f,%.2f,%.2f): ' % (key[0], key[1], key[2]) + str(values)

    for key, value in zip(node.N.keys(), node.N.values()):
        if len(key) == 2:
            N += '(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f):%.2f ' % (key[0][0], key[0][1], key[0][2], key[1][0],
                                                            key[1][1], key[1][2], value)

        else:
            N += '(%.2f,%.2f,%.2f):%d ' % (key[0], key[1], key[2], value)

    if node.parent_action is not None:
        pact = node.parent_action
        if len(pact) == 2:
            parent_action += '(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f)' % (pact[0][0], pact[0][1], pact[0][2], pact[1][0],
                                                                  pact[1][1], pact[1][2])
        else:
            pact = pact.squeeze()
            parent_action += '(%.2f,%.2f,%.2f)' % (pact[0], pact[1], pact[2])
    else:
        parent_action = 'None'
    """

    info = 'parent_action: '+parent_action + '\n' + \
           'N: ' + N + '\n' + \
           'Nvisited: ' + str(node.Nvisited) + '\n' + \
           'Q -' + Q + '\n'+ \
           'R history - ' + sum_rewards_history
    return info


def recursive_write_tree_on_graph(curr_node, graph):
    # todo:
    #   write the object name and whether the place required the CRP planning
    string_form = get_node_info_in_string(curr_node)
    graph.add_node(string_form)
    if curr_node.is_init_node:
        node = graph.get_node(string_form)
        node.attr['color'] = "red"

    if curr_node.is_goal_node:
        node = graph.get_node(string_form)
        node.attr['color'] = "blue"

    for child_idx, child in enumerate(curr_node.children.values()):
        child_string_form = get_node_info_in_string(child)
        graph.add_edge(string_form, child_string_form)  # connect an edge from parent to child
        edge = graph.get_edge(string_form, child_string_form)
        if len(child.parent_action) == 2:
            edge.attr['label'] = 'pick'
        else:
            edge.attr['label'] = 'place'
        recursive_write_tree_on_graph(child, graph)
    return


def write_dot_file(tree, file_idx, suffix):
    print ("Writing dot file..")
    graph = pgv.AGraph(strict=False, directed=True)
    graph.node_attr['shape'] = 'box'
    recursive_write_tree_on_graph(tree.root, graph)
    graph.layout(prog='dot')
    graph.draw('./test_results/mcts_evolutions/'+str(file_idx)+'_'+suffix+'.png')  # draw png
    print ("Done!")


if __name__ == '__main__':
    main()