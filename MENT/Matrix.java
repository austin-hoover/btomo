/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package gov.sns.tools.ment;



import gov.sns.xal.model.probe.traj.*;
import gov.sns.xal.model.probe.*;
import gov.sns.xal.smf.*;
import gov.sns.xal.model.scenario.*;
import gov.sns.xal.model.pvlogger.*;
import gov.sns.xal.model.alg.*;
import gov.sns.xal.smf.data.*;


/**
 *
 * @author tg4
 */

public class Matrix{

    public double ax;
    public double bx;
    public double cx;
    public double dx;

    public double ay;
    public double by;
    public double cy;
    public double dy;

    private Trajectory trajectory;
    

    public Matrix(String sequenceId,final int pvlogid, final double Tkin, final double Trest, final double Charge) {


        PVLoggerDataSource initial_data = new PVLoggerDataSource(pvlogid);
        TransferMapTracker ptracker = new TransferMapTracker();
        AcceleratorSeq sequence = XMLDataManager.loadDefaultAccelerator().getSequence(sequenceId);

        try {if ( sequence != null ) {


     Scenario model = Scenario.newScenarioFor(sequence);

     model.setSynchronizationMode(Scenario.SYNC_MODE_DESIGN);

     Scenario model_pv = initial_data.setModelSource(sequence, model);

     TransferMapProbe tr_map_probe = ProbeFactory.getTransferMapProbe(sequence, ptracker);

     tr_map_probe.setSpeciesRestEnergy(Trest);
     tr_map_probe.setKineticEnergy(Tkin);
     tr_map_probe.setSpeciesCharge(Charge);


     model_pv.setProbe(tr_map_probe);
     model_pv.resync();
     model_pv.run();

     trajectory = model_pv.getTrajectory();


     }} catch (Exception exception) {}


    }


    public void setElemId(final String profileId, final String elemId){
        
         ax = (((MatrixTrajectory)trajectory).getTransferMatrix(profileId,elemId)).getElem(0, 0);
         bx = (((MatrixTrajectory)trajectory).getTransferMatrix(profileId,elemId)).getElem(0, 1);
         cx = (((MatrixTrajectory)trajectory).getTransferMatrix(profileId,elemId)).getElem(1, 0);
         dx = (((MatrixTrajectory)trajectory).getTransferMatrix(profileId,elemId)).getElem(1, 1);

         ay = (((MatrixTrajectory)trajectory).getTransferMatrix(profileId,elemId)).getElem(2, 2);
         by = (((MatrixTrajectory)trajectory).getTransferMatrix(profileId,elemId)).getElem(2, 3);
         cy = (((MatrixTrajectory)trajectory).getTransferMatrix(profileId,elemId)).getElem(3, 2);
         dy = (((MatrixTrajectory)trajectory).getTransferMatrix(profileId,elemId)).getElem(3, 3);

    }


}

