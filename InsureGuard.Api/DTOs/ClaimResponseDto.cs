namespace InsureGuard.Api.DTOs;

public class ClaimResponseDto
{
    public double RiskScore { get; set; }
    public string RiskLevel { get; set; }
    public string Reason { get; set; }

    public int FraudPrediction { get; set; }
    public double FraudProbability { get; set; }
}