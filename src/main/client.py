class Client:

    def __init__(self,
                 customer_id: str,
                 gender: str,
                 senior_citizen: bool,
                 partner: bool,
                 dependents: bool,
                 tenure: int,
                 phone_service: bool,
                 multiple_lines: bool,
                 internet_service: str,
                 online_security: bool,
                 online_backup: bool,
                 device_protection: bool,
                 tech_support: bool,
                 streaming_tv: bool,
                 streaming_movies: bool,
                 contract: str,
                 paperless_billing: bool,
                 payment_method: str,
                 monthly_charges: float,
                 total_charges: float,
                 churn: bool):
        self.customer_id = customer_id
        self.gender = gender
        self.senior_citizen = senior_citizen
        self.partner = partner
        self.dependents = dependents
        self.tenure = tenure
        self.phone_service = phone_service
        self.multiple_lines = multiple_lines
        self.internet_service = internet_service
        self.online_security = online_security
        self.online_backup = online_backup
        self.device_protection = device_protection
        self.tech_support = tech_support
        self.streaming_tv = streaming_tv
        self.streaming_movies = streaming_movies
        self.contract = contract
        self.paperless_billing = paperless_billing
        self.payment_method = payment_method
        self.monthly_charges = monthly_charges
        self.total_charges = total_charges
        self.churn = churn

    def to_string(self):
        separate: str = " | "
        return (self.customer_id + separate
                + self.gender + separate
                + str(self.senior_citizen) + separate
                + str(self.partner) + separate
                + str(self.dependents) + separate
                + str(self.tenure) + separate
                + str(self.phone_service) + separate
                + str(self.multiple_lines) + separate
                + self.internet_service + separate
                + str(self.online_security) + separate
                + str(self.online_backup) + separate
                + str(self.device_protection) + separate
                + str(self.tech_support) + separate
                + str(self.streaming_tv) + separate
                + str(self.streaming_movies) + separate
                + self.contract + separate
                + str(self.paperless_billing) + separate
                + self.payment_method + separate
                + str(self.monthly_charges) + separate
                + str(self.total_charges) + separate
                + str(self.churn) + separate)
