from anytree import Node, PreOrderIter, findall
import random
import re
import copy
import pandas as pd


def parse_to_tree(lrml: str):
    node_id = -1
    root_node = Node('root', node_id=node_id)
    new_node = None
    current_node = root_node
    current_word = ''
    quote = False
    for i in lrml:
        if i == "'":
            quote = not quote
            current_word += i
        elif not quote and (i == '(' or i == ')' or i == ','):
            if current_word:
                node_id += 1
                new_node = Node(current_word, current_node, node_id=node_id)
                current_word = ''
            if i == '(' and new_node is not None:
                current_node = new_node
            elif i == ')' and current_node.parent is not None:
                current_node = current_node.parent
        else:
            current_word += i
    if current_node == root_node and current_word:
        node_id += 1
        new_node = Node(current_word, current_node, node_id=node_id)
    return root_node


def node_to_lrml(node, stop_node=None, separator=','):
    initial_depth = node.depth
    last_depth = -1
    lrml = ''
    for i in PreOrderIter(node):
        if i.depth > last_depth:
            if last_depth != -1:
                lrml += '('
        else:
            last_depth - i.depth
            lrml += ')' * (last_depth - i.depth)
            lrml += separator
        if stop_node is not None and i.node_id == stop_node.node_id:
            break
        lrml += i.name
        last_depth = i.depth
    #   Only add brackets for full print
    if stop_node is None:
        lrml += ')' * (last_depth - initial_depth)
    #   Remove root node
    if node.is_root:
        lrml = lrml.replace('root(', '')
        if stop_node is None:
            lrml = lrml[:-1]
    return lrml


def get_auto_completion_training_samples(lrml, number):
    tree = parse_to_tree(lrml)
    return [get_auto_completion_pair(tree) for i in range(number)]


def get_auto_completion_pair(tree):
    choice = random.choice(tree.descendants)
    return node_to_lrml(tree, stop_node=choice), node_to_lrml(choice)



def add_spaces_to_lrml(lrml):
    return lrml.str.replace(r'([a-z0-9])([A-Z%])', r'\1_\2', regex=True).str.lower().str.replace(
        '_', ' ', regex=False).str.replace(
        '(', '( ', regex=False).str.replace(
        ',', ', ', regex=False).str.replace(
        r' +', ' ', regex=True).str.replace(
        ' ,', ',', regex=False)


def replace_lrml_schema_terms(lrml):
    return lrml.str.replace('base unit(', 'baseunit(', regex=False).str.replace('fun(', 'function(',
                                                                                regex=False).str.replace('var(',
                                                                                                         'variable(',
                                                                                                         regex=False).str.replace(
        'rel(', 'relation(', regex=False).str.replace('expr(', 'expression(', regex=False)


def reverse_lrml_schema_terms(lrml):
    return lrml.str.replace('function(', 'fun(', regex=False).str.replace('variable(', 'var(', regex=False).str.replace(
        'relation(', 'rel(', regex=False).str.replace('expression(', 'expr(', regex=False)


def fix_lrml_tokenisation(lrml):
    return replace_lrml_schema_terms(add_spaces_to_lrml(lrml))


def new_or_old_if_na_df(df, new, old):
    return df.apply(lambda x: new_or_old_if_na_row(x, new, old), axis=1)


def new_or_old_if_na_row(row, new, old):
    if not pd.isna(row[new]):
        return row[new]
    return row[old]


def normalise_text(text):
    text = text.strip()
    if text[-1] != '.':
        text += '.'
    return text


# fix_lrml_tokenisation on a string level
def token_fix(lrml):
    return re.sub(r' +', ' ', re.sub(r'([a-z0-9])([A-Z%])', r'\1_\2', lrml).replace(
        '_', ' ').replace(
        '(', '( ').replace(
        ',', ', ')).replace(
        ' ,', ',').replace(
        'base unit(', 'baseunit(').replace(
        'fun(', 'function(').replace(
        'var(', 'variable(').replace(
        'rel(', 'relation(').replace(
        'expr(', 'expression(').lower()

def remove_duplicate_and_or_expr(lrml):
    tree = parse_to_tree(lrml)
    and_node = findall(tree, filter_=lambda x: (x.name == 'and' and x.parent.name == 'and') or (
        x.name == 'or' and x.parent.name == 'or') or (x.name == 'expr' and x.parent.name == 'expr' and len(x.siblings) == 0))
    while and_node:
        parent = and_node[0].parent
        and_node[0].parent = None
        for i in and_node[0].children:
            i.parent = parent
        parent.children = sorted(
            parent.children, key=lambda item: item.node_id)
        and_node = findall(tree, filter_=lambda x: (x.name == 'and' and x.parent.name == 'and') or (
            x.name == 'or' and x.parent.name == 'or') or (x.name == 'expr' and x.parent.name == 'expr' and len(x.siblings) == 0))
    return node_to_lrml(tree)


def sort_children(node):
    node.children = sorted(node.children, key=lambda item: item.node_id)


def remove_node(node):
    if node.parent is None:
        return
    parent = node.parent
    node.parent = None
    for i in node.children:
        i.parent = parent
    sort_children(parent)


def combine_rel_and_var(lrml):
    tree = parse_to_tree(lrml)
    expr_node = findall(tree, filter_=lambda x: (x.name == 'atom' in str(x)))

    for i in expr_node:
        if len(i.children) == 2 and i.children[0].name.strip() in ['rel', 'relation'] and i.children[1].name.strip() in ['var', 'variable']:
            new_name = i.children[1].children[0].name + \
                '.' + i.children[0].children[0].name
            i.children[1].parent = None
            i.children[0].name = new_name
            for j in i.children[0].children:
                j.parent = None
        elif len(i.children) == 1:
            new_name = i.children[0].children[0].name
            i.children[0].name = new_name
            for j in i.children[0].children:
                j.parent = None
        else:
            print('Error: ', i.children)
    return node_to_lrml(tree)


def resolve_expressions(lrml):
    tree = parse_to_tree(lrml)
    expr_node = findall(tree, filter_=lambda x: (
        x.name.strip() in ['expr', 'expression']))
    for i in expr_node:
        funs = findall(i, maxlevel=2, filter_=lambda x: (
            x.name.strip() in ['fun', 'function']))
        if len(funs) != 1:
            if not i.children or not i.children[0].name == 'rulestatement':
                print('Error: ', i, i.children, funs, lrml)
        else:
            fun = funs[0].children[0]
            funs[0].parent = None
            fun.parent = i.parent
            i.parent = None
            for j in i.children:
                if len(j.children) == 1:
                    j.parent = None
                    j.children[0].node_id = j.node_id
                    j.children[0].parent = fun
                else:
                    j.parent = fun
            sort_children(fun.parent)
    return node_to_lrml(tree)


def resolve_loop(lrml):
    tree = parse_to_tree(lrml)
    expr_node = findall(tree, filter_=lambda x: (
        (x.name.strip() in ['expr', 'expression'])))
    for i in expr_node:
        if not len(i.children) == 2:
            print(i, i.children)
        else:
            for j in i.children:
                remove_node(j)
            i.name = 'loop'
    return node_to_lrml(tree)


abbr_mapping = {'metre': 'm', 'gram': 'g', 'litre': 'l', 'newton': 'N',
                'pascal': 'Pa', 'angleDegree': 'deg', 'celsius': 'degC', 'hectare': 'ha'}
abbr_mapping_prefixes = {'kilo': 'k', 'milli': 'm', 'mega': 'M'}
opperator_mapping = {'addition': '+', 'subtraction': '-',
                     'multiplication': '*', 'division': '/'}


def resolve_units(lrml):
    tree = parse_to_tree(lrml)
    unit_nodes = findall(tree, filter_=lambda x: (((x.name == 'baseunit') and (
        x.parent.name != 'derivedunit')) or (x.name == 'derivedunit')))

    for node in unit_nodes:
        if node.name == 'baseunit':
            unit = base_unit_to_abbr(node)
        else:
            unit = derived_unit_to_abbr(node)
        value_node = findall(
            node.parent, filter_=lambda x: (x.name == 'value'))[0]
        value = trim_decimal_point(value_node.leaves[0].name)
        Node(name=value + ' ' + unit, parent=node.parent)
        node.parent = None
        value_node.parent = None

    return node_to_lrml(tree)


def trim_decimal_point(number_string):
    return number_string[:-2] if number_string.endswith('.0') else number_string


def base_unit_to_abbr(node):
    abbreviations = [''] * 5
    for child in node.children:
        abbreviation = ''
        if child.name == 'prefix':
            abbreviations[0] = abbr_mapping_prefixes[child.leaves[0].name]
        elif child.name == 'kind':
            abbreviations[1] = abbr_mapping[child.leaves[0].name]
        elif child.name == 'exponent':
            exp = child.leaves[0].name
            abbreviations[2] = trim_decimal_point(exp)
        else:
            print('Unknown child', child.name)

    return ''.join(abbreviations)


def derived_unit_to_abbr(node):
    abbreviations = [''] * 3
    counter = 0

    for child in node.children:
        if child.name == 'baseunit':
            abbreviations[counter] = base_unit_to_abbr(child)
            counter += 2
        elif child.name == 'operator':
            abbreviations[1] = opperator_mapping[child.leaves[0].name]
        else:
            print('Unknown child', child.name)
    return ''.join(abbreviations)


regex = r'(?<!\w)(\d+\.?\d*)\s([a-zA-Z0-9/*+-]+)(?!\w)'


def reverse_baseunit(value, prefix):
    if not value:
        return None
    base_unit = Node(name=prefix + 'baseunit')
    exp = None
    prefix_node = None
    kind = None
    # Ends with number -> exponent
    if re.search(r'\d+$', value):
        exp = Node(name=prefix + 'exponent')
        Node(name=prefix + value[-1] + '.0', parent=exp)
        value = value[:-1]
    # Ends with abbr -> kind
    for abbr, unit in reversed(abbr_mapping.items()):
        if value.endswith(unit):
            kind = Node(name=prefix + 'kind')
            Node(name=prefix + abbr, parent=kind)
            value = value[:-len(unit)]
            break
    # Remainder -> prefix
    for abbr, unit in reversed(abbr_mapping_prefixes.items()):
        if value == unit:
            prefix_node = Node(name=prefix + 'prefix')
            Node(name=prefix + abbr, parent=prefix_node)
            value = ''
            break

    base_unit.children = [i for i in [exp, prefix_node, kind] if i]

    if value != '':
        return None
    return base_unit


def reverse_units(lrml, prefix=' '):
    tree = parse_to_tree(lrml)
    data_nodes = findall(tree, filter_=lambda x: ((x.name.strip() == 'data')))
    for i in data_nodes:
        data_value = i.leaves[0].name.strip()
        if re.match(regex, data_value):
            if len(data_value.split(' ')) > 2:
                continue
            number = data_value.split(' ')[0]
            unit = data_value.split(' ')[1]
            unit_node = None
            # Derived Unit
            for operator_name, operator in opperator_mapping.items():
                split_unit = unit.split(operator)
                if len(split_unit) == 2:
                    first_unit = reverse_baseunit(split_unit[0], prefix=prefix)
                    second_unit = reverse_baseunit(
                        split_unit[1], prefix=prefix)
                    operator = Node(name=prefix + 'operator')
                    name_node = Node(name=prefix + 'name', parent=operator)
                    Node(name=operator_name, parent=name_node)
                    if first_unit and second_unit:
                        unit_node = Node(name=prefix + 'derivedunit', children=[
                                         first_unit, operator, second_unit])
                    break
            # Base unit
            if not unit_node:
                unit_node = reverse_baseunit(unit, prefix=prefix)

            if unit_node:
                value_node = Node(name=prefix + 'value')
                if not '.' in number:
                    number += '.0'

                Node(name=prefix + number, parent=value_node)
                i.children = [unit_node, value_node]

    return node_to_lrml(tree)


def get_node_path(node):
    path = ''
    while node:
        path = '/' + node.name + path
        node = node.parent
    return path


def find_non_data_leave_names(node):
    return [i.name for i in node.leaves if '/data/' not in str(i)]


def find_first_non_data_leave(node):
    return [i for i in node.leaves if '/data/' not in str(i)][0]


def find_data_node(node):
    data_node = [i for i in node.leaves if '/data/' in str(i)]
    assert len(data_node) == 1, (node, data_node)
    data_node = data_node[0]
    while data_node.name != 'data':
        data_node = data_node.parent
    return data_node


def find_next_and_or_node(node):
    while node and node.name != 'and' and node.name != 'or':
        node = node.parent
    return node


def find_max_express_node(node, include_deonitics=True):
    outside_options = ['not', 'expression', 'expr']
    if include_deonitics:
        outside_options += ['obligation', 'permission', 'prohibition']
    first_apearance = node.name.strip() in outside_options
    current_apperance = node.name.strip() in outside_options
    while node.parent:
        first_apearance += node.parent.name.strip() in outside_options
        current_apperance = node.parent.name.strip() in outside_options
        if node and (not first_apearance or (first_apearance and current_apperance)):
            node = node.parent
        else:
            break
    return node


def increase_indices(nodes, start_index, increase):
    for node in nodes:
        if node.node_id >= start_index:
            node.node_id += increase


def move_and_or_to_data_node(lrml):
    tree = parse_to_tree(lrml)
    and_nodes = findall(tree, filter_=lambda x: (
        (x.name == 'and') or (x.name == 'or')))
    for and_node in and_nodes:
        expr_nodes = list(dict.fromkeys([find_max_express_node(i) for i in findall(
            and_node, filter_=lambda x: ((x.name == 'expr')))]))
        # Find all expr nodes with the same parent and same children
        indices = []
        for index, expr_node in enumerate(expr_nodes):
            if index in indices:
                continue
            and_node = find_next_and_or_node(expr_node)
            similar_nodes = [expr_node]
            for index2, expr_node2 in enumerate(expr_nodes[index+1:]):
                if expr_node.parent and expr_node2.parent and expr_node.node_id != expr_node2.node_id and expr_node.parent.name == expr_node2.parent.name != 'expr' and \
                        expr_node.name == expr_node2.name and find_non_data_leave_names(expr_node) == find_non_data_leave_names(expr_node2) and \
                        find_next_and_or_node(expr_node) == find_next_and_or_node(expr_node2) and \
                        get_node_path(find_first_non_data_leave(expr_node)) == get_node_path(find_first_non_data_leave(expr_node2)):
                    similar_nodes.append(expr_node2)
                    indices.append(index+index2)
                else:
                    # Only for neighbouring nodes
                    break

            try:
                if len(similar_nodes) > 1:
                    data_nodes = [find_data_node(i) for i in similar_nodes]
                    for i in data_nodes:
                        assert len(i.children) == 1, (i, i.children)

                    new_and_node = Node(and_node.name, parent=data_nodes[0])

                    for similar_node in similar_nodes[1:]:
                        similar_node.parent = None
                        i.parent = new_and_node
                    new_and_node.children = [i.children[0] for i in data_nodes]
            except AssertionError:
                print('Cannot combine baseunit and value with other data')
            if and_node and len(and_node.children) == 1:
                remove_node(and_node)
    return node_to_lrml(tree)


def increase_indices(nodes, start_index, increase):
    for node in nodes:
        if node.node_id >= start_index:
            node.node_id += increase


def reverse_move_and_or_to_data_node(lrml):
    tree = parse_to_tree(lrml)
    and_nodes = findall(tree, filter_=lambda x: (
        (x.name.strip() == 'and') or (x.name.strip() == 'or')))
    for and_node in and_nodes:
        if and_node.parent.name.strip() == 'data':
            connector_name = and_node.name
            copyable_node = find_max_express_node(and_node)
            copied_nodes = []
            children = and_node.children
            data_node = and_node.parent
            and_node.parent = None
            for index, child in enumerate(children):
                new_node = copy.deepcopy(copyable_node)
                child.parent = findall(
                    new_node, filter_=lambda x: str(x) == str(data_node))[0]
                new_node.node_id += index
                copied_nodes.append(new_node)

            parent_node = copyable_node.parent
            if parent_node:
                if parent_node.name.strip() == connector_name.strip():
                    increase_indices(copyable_node.siblings,
                                     copyable_node.node_id, len(children))
                    parent_node.children = parent_node.children[:copyable_node.node_id] + tuple(
                        copied_nodes) + parent_node.children[copyable_node.node_id:]
                    sort_children(parent_node)
                else:
                    connector_node = Node(
                        connector_name, parent=parent_node, node_id=copyable_node.node_id)
                    connector_node.children = copied_nodes
                    sort_children(connector_node.parent)
                copyable_node.parent = None

    return node_to_lrml(tree)


keywords = ['if', 'then', 'and', 'or', 'obligation', 'permission', 'prohibition', 'not', 'expression', 'appliedstatement', 'rulestatement',
            'atom', 'function', 'relation', 'variable', 'data', 'baseunit', 'derivedunit', 'prefix', 'kind', 'operator', 'value']

# This function tries to revert the simplified LRML back to the original LRML by adding the missing keywords
# The first nodes that are no keywords are assumed to be expressions and the node names are the new functions of this expression
# For the children there are multiple possibilities: The first can be either an atom or a new expression. The second one, if available, is always a data node


def reverse_resolve_expressions(lrml, fix_errors, prefix=' '):
    tree = parse_to_tree(lrml)
    recusive_reverse_resolve_expressions(tree, fix_errors, prefix)
    return node_to_lrml(tree)


def recusive_reverse_resolve_expressions(node, fix_errors, prefix):
    for node in node.children:
        if node.name.strip() in keywords:
            recusive_reverse_resolve_expressions(
                node, fix_errors, prefix=prefix)
        else:
            make_expression(node, fix_errors, prefix=prefix)


def make_expression(node, fix_errors, prefix):
    expr = Node(prefix + 'expression',
                parent=node.parent, node_id=node.node_id)
    fun = Node(prefix + 'function', parent=expr, node_id=node.node_id)
    node.parent = fun
    children = node.children
    if children:
        if children[0].children:
            children[0].parent = expr
            make_expression(children[0], fix_errors, prefix=prefix)
        else:
            atom = Node(prefix + 'atom', parent=expr,
                        node_id=children[0].node_id)
            children[0].parent = atom
        if len(children) > 1:
            if children[1].name.strip() == 'data':
                children[1].parent = expr
            else:
                data = Node(prefix + 'data', parent=expr,
                            node_id=children[1].node_id)
                children[1].parent = data
        if node.children:
            if fix_errors:
                for child in node.children:
                    child.parent = None
        sort_children(expr)
        sort_children(expr.parent)
    else:
        print('ERROR: No children for expression', node_to_lrml(node.root))


def reverse_combine_rel_and_var(lrml, prefix=' '):
    tree = parse_to_tree(lrml)
    atom_node = findall(tree, filter_=lambda x: (
        x.name == prefix + 'atom' in str(x)))

    for i in atom_node:
        if i.children:
            nodes = re.split(
                '(?=[a-z' + prefix + '])\.|\.(?=[a-z' + prefix + '])', i.children[0].name)
            if len(nodes) > 1:
                rel = nodes[1]
                rel_node = Node(name=prefix + 'relation', parent=i)
                Node(name=rel, parent=rel_node)
            var = nodes[0]
            var_node = Node(name=prefix + 'variable', parent=i)
            Node(name=var, parent=var_node)
            i.children[0].parent = None

    return node_to_lrml(tree)


def reverse_loop(lrml, prefix=' '):
    tree = parse_to_tree(lrml)
    hierarchy = ['rulestatement', 'appliedstatement']
    loop_node = findall(tree, filter_=lambda x: (
        x.name == prefix + 'loop' in str(x)))
    for i in loop_node:
        i.name = prefix + 'expression'
        for index, j in enumerate(i.children):
            if index > 1:
                print('ERROR in reverse loop:', [
                      node_to_lrml(i) for i in i.children])
                break
            node = Node(prefix + hierarchy[index], parent=i, node_id=j.node_id)
            j.parent = node
            sort_children(i)

    return node_to_lrml(tree)


def get_leave_names_from_node(node):
    return [i.name for i in node.leaves]


def remove_duplicate_expressions(lrml, keyword):
    tree = parse_to_tree(lrml)
    nodes = findall(tree, filter_=lambda x: (x.name == keyword in str(x)))
    for i in range(len(nodes) - 1):
        for j in range(i+1, len(nodes)):
            if nodes[i].parent == nodes[j].parent and get_leave_names_from_node(nodes[i]) == get_leave_names_from_node(nodes[j]):
                nodes[j].parent = None
    return node_to_lrml(tree)


expression_map = {'fun': 'function', 'var': 'variable',
                  'rel': 'relation', 'expr': 'expression'}


def tree_based_spacing(lrml):
    tree = parse_to_tree(lrml)
    regex = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
    regex2 = re.compile(r'(?<=[a-z])\.(?=[a-z])')
    for i in PreOrderIter(tree):
        if i.children:
            if i.name in expression_map:
                i.name = expression_map[i.name]
            i.name = regex.sub(' ', i.name).lower()
            i.name = ' ' + i.name
        else:
            if not ' ' in i.name:
                i.name = regex.sub(' ', i.name).lower().replace('_', ' ')
                i.name = regex2.sub('. ', i.name).replace('_', ' ')
            else:
                i.name = i.name.replace('_', ' ')
            i.name = ' ' + i.name

    return node_to_lrml(tree).strip()
