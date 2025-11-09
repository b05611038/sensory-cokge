from sensory_cokge.models import (BERT_embeddings,
                                  Laser_embeddings)

def main():
    brewing_malt_context = 'This brewing malt has {0} {1} odor.'
    descriptions = ['honey', 'corn']
    brewing_malt_embeddings = BERT_embeddings(descriptions, context = brewing_malt_context)
    for idx in brewing_malt_embeddings:
        print(brewing_malt_embeddings[idx]['description'] + ':', brewing_malt_embeddings[idx]['encoder_embedding'])

    berry_context = 'This blueberry has {0} {1} taste.'
    descriptions = ['grass', 'minty']
    berry_embeddings = Laser_embeddings(descriptions, context = berry_context)
    for idx in berry_embeddings:
        print(berry_embeddings[idx]['description'] + ':', berry_embeddings[idx]['encoder_embedding'])

    print('Demo done.')

    return None

if __name__ == '__main__':
    main()


