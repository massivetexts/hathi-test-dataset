
def already_imported_list():
    return set([l.rstrip() for l in open("data_outputs/completed.csv")])
