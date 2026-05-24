!!! Note ""

    The accuracy of the POS tagger in the current version is {{ posTagger_evaluation_value }}%.

::: hazm.pos_tagger
    handler: python
    options:
        members:
            - POSTagger
            - SpacyPOSTagger
            - StanfordPOSTagger
        show_root_heading: false
        show_source: false
