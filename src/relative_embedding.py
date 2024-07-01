import torch

from .distances import cosine_similarity

__all__ = ['construct_relative_embeddings', 'RelativeEmbedding', 'DescriptorColors',
        'AnchorDescriptors']

AnchorDescriptors = ['floral', 'fruity', 'sour/fermented', 'green/vegetative', 'other',
            'roasted', 'spices', 'nutty/cocoa', 'sweet']

DescriptorColors = {
        'floral': (209, 30, 92),
        'fruity': (210, 39, 31),
        'sour/fermented': (231, 174, 27),
        'green/vegetative': (31, 110, 44),
        'other': (28, 151, 172),
        'roasted': (191, 69, 42),
        'spices': (161, 37, 53),
        'nutty/cocoa': (157, 113, 89),
        'sweet': (223, 84, 44),
}

def construct_relative_embeddings(model, source_struc, anchor_descirptions, embedding_properties):
    assert source_struc in ['encoder', 'decoder']
    
    target_embedding = '{0}_embedding'.format(source_struc)
    anchor_embeddings, embedding_size = {}, -1
    for anchor_name in anchor_descirptions:
        for idx in embedding_properties:
            if anchor_name == embedding_properties[idx]['description']:
                anchor_embeddings[anchor_name] = embedding_properties[idx][target_embedding]
                break
    
    assert len(anchor_descirptions) == len(anchor_embeddings)
    for anchor_name in anchor_embeddings:
        embedding_size = max(anchor_embeddings[anchor_name].shape[0], embedding_size)

    embedding_projector = RelativeEmbedding('{0} ({1})'.format(model, source_struc),
                                            embedding_size,
                                            anchor_descirptions,
                                            anchor_embeddings)

    embedding_name = 'relative_{0}_embedding'.format(source_struc)
    for idx in embedding_properties:
        description_embedding = embedding_properties[idx][target_embedding]
        relative_description_embedding = embedding_projector(description_embedding)
        embedding_properties[idx][embedding_name] = relative_description_embedding

    return embedding_properties

class RelativeEmbedding:
   def __init__(self, model, embedding_size, anchor_names, anchor_embeddings):
       assert type(model) == str
       self.model = model

       assert type(embedding_size) == int
       self.embedding_size = embedding_size

       assert type(anchor_names) == list
       assert type(anchor_embeddings) == dict
       assert len(anchor_names) == len(anchor_embeddings)
       self.anchor_names = anchor_names
       self.anchor_embeddings = anchor_embeddings

   def __repr__(self):
       return "{0}(model='{1}', anchors={2})".format(self.__class__.__name__,
                                                     self.model,
                                                     self.anchor_names)

   def __call__(self, embedding):
       assert embedding.shape[0] == self.embedding_size

       relative_embedding = [cosine_similarity(embedding, self.anchor_embeddings[anchor_name]) \
               for anchor_name in self.anchor_names]

       return torch.tensor(relative_embedding, dtype = torch.float32) 


