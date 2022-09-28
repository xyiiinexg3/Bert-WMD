class CCKSDataset:
    """
    Dataset which stores the tweets and returns them as processed features
    """

    def __init__(self, query, question, label=None):
        # self.query = query
        # self.question = question
        # self.label = label
        # self.tokenizer = config.TOKENIZER
        # self.max_len = config.MAX_LEN
        file = open('')

    def __len__(self):
        return len(self.query)

    def __getitem__(self, item):
        query = self.query[item]
        question = self.question[item]
        query_ids, query_mask, query_type_ids = self.process_data(query)
        question_ids, question_mask, question_type_ids = self.process_data(question)

        if self.label is not None:
            label = self.label[item]
            # Return the processed data where the lists are converted to `torch.tensor`s
            return {
                "query": query,
                "question": question,
                'query_ids': torch.tensor(query_ids, dtype=torch.long),
                'query_mask': torch.tensor(query_mask, dtype=torch.long),
                'query_type_ids': torch.tensor(query_type_ids, dtype=torch.long),
                'question_ids': torch.tensor(question_ids, dtype=torch.long),
                'question_mask': torch.tensor(question_mask, dtype=torch.long),
                'question_type_ids': torch.tensor(question_type_ids, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'query_ids': torch.tensor(query_ids, dtype=torch.long),
                'query_mask': torch.tensor(query_mask, dtype=torch.long),
                'token_type_ids': torch.tensor(query_type_ids, dtype=torch.long),
                'question_ids': torch.tensor(question_ids, dtype=torch.long),
                'question_mask': torch.tensor(question_mask, dtype=torch.long),
                'question_type_ids': torch.tensor(question_type_ids, dtype=torch.long),
            }

    def process_data(self, q):
        input_ids = self.tokenizer.encode(q).ids

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:-1][:self.max_len - 1] + [102]

        token_type_ids = [0] * len(input_ids)
        input_masks = [1] * len(token_type_ids)

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            input_masks = input_masks + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len
        assert len(token_type_ids) == self.max_len

        return input_ids, input_masks, token_type_ids