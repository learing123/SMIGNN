smignn_conf = {
    'model_list': ['social', 'rating', 'fusion'],
    'distill_dict': {
        ('social', 'rating'): 0.0,
        ('rating', 'social'): 0.0,
        ('social', 'fusion'): 0.0,
        ('fusion', 'social'): 0.0,
        ('rating', 'fusion'): 0.0,
        ('fusion', 'rating'): 0.0,
    },
    'l_user': [1, 2],
    'l_item': [0],
}