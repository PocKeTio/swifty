using System;
using System.Collections.Generic;

namespace SwiftAnalyzer.Core.Models
{
    public class Swift799Message
    {
        public string OriginalText { get; set; } = "";
        public string MainTask { get; set; } = "";
        public bool HasConditionalActions { get; set; }
        public List<SubTask> SubTasks { get; set; } = new();
        public List<TimeConstraint> TimeConstraints { get; set; } = new();
    }

    public class TimeConstraint
    {
        public string Type { get; set; } = "";
        public float Confidence { get; set; }
    }

    public class SubTask
    {
        public string Type { get; set; } = "";
        public string Text { get; set; } = "";
        public float Confidence { get; set; }
        public List<(string Label, float Score)> RelatedTasks { get; set; } = new();
    }
}
