namespace InsureGuard.Api.DTOs;

public class ClaimRequestDto
{
    public double total_claim_amount { get; set; }
    public double injury_claim { get; set; }
    public double vehicle_claim { get; set; }
    public double property_claim { get; set; }

    public string incident_type { get; set; }
    public string incident_severity { get; set; }
    public string collision_type { get; set; }
    public string police_report_available { get; set; }

    public int incident_hour_of_day { get; set; }
    public int claim_sequence { get; set; }
}