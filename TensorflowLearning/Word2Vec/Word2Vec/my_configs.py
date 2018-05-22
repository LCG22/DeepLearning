#! /usr/bin/env python
# coding: utf-8


class Public(object):
    
    def __init__(self):
        # 停用词
        self.stop_words = [
            u"｛", u"，", u"｝", u"……", u"!", u"#", u'"',  u"%",  u"$",  u"'",  u"&",  u")",  u"(",  u"+",  u"*",
            u"-",  u",",  u"/",  u".",  u"￥",  u"》",  u";",  u":",  u"=",  u"<",  u"?",  u">",  u"@",  u"——",
            u"。",  u"：",  u"；",  u"？",  u"！",  u"（",  u"）",  u"[",  u"]",  u"\\",  u"_",  u"^",  u"`",  u"   ",
            u"【",  u"】",  u"”",  u"“",  u"’",  u"《",  u"·",  u"‘",  u"、",  u"{",  u"}",  u"|",  u"~",  u" ",
            u""
        ]
    
    
class ProcessingWord(Public):
    
    def __init__(self):
        super(ProcessingWord, self).__init__()
        
        
class TrainModel(Public):
    
    def __init__(self):
        super(TrainModel, self).__init__()